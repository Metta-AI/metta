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
// Used for utf32 -> utf8 conversion, which is needed for reading the map.
// Hopefully this is temporary.
#include <codecvt>
#include <locale>

namespace py = pybind11;

MettaGrid::MettaGrid(py::dict env_cfg, py::array map) {
  // env_cfg is a dict-form of the OmegaCong config.
  // `map` is a numpy array of shape (height, width) and a 'U' dtype.
  // It's not clear that there's any value in using a numpy array here,
  // and this causes us to need to do some awkward conversions.
  auto cfg = env_cfg["game"].cast<py::dict>();
  _cfg = cfg;

  int num_agents = cfg["num_agents"].cast<int>();
  _max_timestep = cfg["max_steps"].cast<unsigned int>();
  _obs_width = cfg["obs_width"].cast<unsigned short>();
  _obs_height = cfg["obs_height"].cast<unsigned short>();

  _current_timestep = 0;

  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  auto map_info = map.request();

  _grid = std::make_unique<Grid>(map_info.shape[1], map_info.shape[0], layer_for_type_id);
  _obs_encoder = std::make_unique<ObservationEncoder>();
  _grid_features = _obs_encoder->feature_names();

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>();

  _event_manager->init(_grid.get(), _stats.get());
  _event_manager->event_handlers.push_back(new ProductionHandler(_event_manager.get()));
  _event_manager->event_handlers.push_back(new CoolDownHandler(_event_manager.get()));

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

  // Ensure the array is 2D
  if (map_info.ndim != 2) {
    throw std::runtime_error("Expected a 2D array");
  }

  // Check the dtype. utf-32 is 4 bytes.
  ssize_t element_size = map_info.itemsize;
  if (element_size % 4 != 0) {
    throw std::runtime_error("Invalid Unicode size: not divisible by 4");
  }
  ssize_t max_chars = element_size / 4;

  // Initialize objects from map
  auto map_data = static_cast<wchar_t*>(map_info.ptr);
  for (int r = 0; r < map_info.shape[0]; r++) {
    for (int c = 0; c < map_info.shape[1]; c++) {
      ssize_t idx = r * map_info.shape[1] + c;
      Converter* converter = nullptr;
      std::wstring wide_string(map_data + idx * max_chars, max_chars);
      std::wstring_convert<std::codecvt_utf8<wchar_t>> w_string_converter;
      std::string cell = w_string_converter.to_bytes(wide_string);
      cell = cell.substr(0, cell.find('\0'));
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

  // Initialize buffers. The buffers are likely to be re-set by the user anyways,
  // so nothing above should depend on them before this point.
  std::vector<ssize_t> shape = {static_cast<ssize_t>(num_agents),
                                static_cast<ssize_t>(_obs_height),
                                static_cast<ssize_t>(_obs_width),
                                static_cast<ssize_t>(_grid_features.size())};
  auto observations = py::array_t<unsigned char, py::array::c_style>(shape);
  auto terminals = py::array_t<bool, py::array::c_style>(static_cast<ssize_t>(num_agents));
  auto truncations = py::array_t<bool, py::array::c_style>(static_cast<ssize_t>(num_agents));
  auto rewards = py::array_t<float, py::array::c_style>(static_cast<ssize_t>(num_agents));

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
                                     size_t agent_idx) {
  auto observation_view = _observations.mutable_unchecked<4>();

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
  for (unsigned int r = r_start; r < r_end; r++) {
    for (unsigned int c = c_start; c < c_end; c++) {
      for (unsigned int layer = 0; layer < _grid->num_layers; layer++) {
        GridLocation object_loc(r, c, layer);
        auto obj = _grid->object_at(object_loc);
        if (!obj) continue;

        int obs_r = object_loc.r + obs_height_radius - observer_row;
        int obs_c = object_loc.c + obs_width_radius - observer_col;

        auto agent_obs = observation_view.mutable_data(agent_idx, obs_r, obs_c, 0);
        _obs_encoder->encode(obj, agent_obs);
      }
    }
  }
}

void MettaGrid::_compute_observations(py::array_t<int> actions) {
  for (size_t idx = 0; idx < _agents.size(); idx++) {
    auto& agent = _agents[idx];
    _compute_observation(agent->location.r, agent->location.c, _obs_width, _obs_height, idx);
  }
}

void MettaGrid::_step(py::array_t<int> actions) {
  auto actions_view = actions.unchecked<2>();

  // Reset rewards and observations
  auto rewards_view = _rewards.mutable_unchecked<1>();

  std::fill(
      static_cast<float*>(_rewards.request().ptr), static_cast<float*>(_rewards.request().ptr) + _rewards.size(), 0);

  auto obs_ptr = static_cast<unsigned char*>(_observations.request().ptr);
  auto obs_size = _observations.size();
  std::fill(obs_ptr, obs_ptr + obs_size, 0);

  std::fill(_action_success.begin(), _action_success.end(), false);

  // Increment timestep and process events
  _current_timestep++;
  _event_manager->process_events(_current_timestep);

  // Process actions by priority
  for (unsigned char p = 0; p <= _max_action_priority; p++) {
    for (size_t idx = 0; idx < _agents.size(); idx++) {
      int action = actions_view(idx, 0);
      if (action < 0 || action >= _num_action_handlers) {
        printf("Invalid action: %d\n", action);
        continue;
      }

      ActionArg arg = actions_view(idx, 1);
      auto& agent = _agents[idx];
      auto& handler = _action_handlers[action];

      if (handler->priority != _max_action_priority - p) {
        continue;
      }

      if (arg > _max_action_args[action]) {
        continue;
      }

      _action_success[idx] = handler->handle_action(idx, agent->id, arg, _current_timestep);
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
  if (_max_timestep > 0 && _current_timestep >= _max_timestep) {
    std::fill(static_cast<bool*>(_truncations.request().ptr),
              static_cast<bool*>(_truncations.request().ptr) + _truncations.size(),
              1);
  }
}

py::tuple MettaGrid::reset() {
  if (_current_timestep > 0) {
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
  auto obs_ptr = static_cast<unsigned char*>(_observations.request().ptr);
  auto obs_size = _observations.size();
  std::fill(obs_ptr, obs_ptr + obs_size, 0);

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
  {
    auto observation_info = _observations.request();
    auto shape = observation_info.shape;
    if (observation_info.ndim != 4 || shape[0] != num_agents || shape[1] != _obs_height || shape[2] != _obs_width ||
        shape[3] != _grid_features.size()) {
      std::stringstream ss;
      ss << "observations has shape [" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3]
         << "] but expected [" << num_agents << ", " << _obs_height << ", " << _obs_width << ", "
         << _grid_features.size() << "]";
      throw std::runtime_error(ss.str());
    }
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

void MettaGrid::set_buffers(py::array_t<unsigned char, py::array::c_style>& observations,
                            py::array_t<bool, py::array::c_style>& terminals,
                            py::array_t<bool, py::array::c_style>& truncations,
                            py::array_t<float, py::array::c_style>& rewards) {
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _episode_rewards = py::array_t<float>(_rewards.shape(0));
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards.mutable_unchecked<1>()(i));
  }

  validate_buffers();
}

py::tuple MettaGrid::step(py::array_t<int> actions) {
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

    // Get feature offsets for this object type
    auto type_features = _obs_encoder->type_feature_names()[obj->_type_id];
    std::vector<uint8_t> offsets(type_features.size());
    // We shouldn't have more than 256 features, since we're storing the feature_ids
    // as uint_8ts.
    assert(offsets.size() < 256);
    for (uint8_t i = 0; i < offsets.size(); i++) {
      offsets[i] = i;
    }
    unsigned char obj_data[type_features.size()];

    // Encode object features
    _obs_encoder->encode(obj, obj_data, offsets);

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

unsigned int MettaGrid::current_timestep() {
  return _current_timestep;
}

unsigned int MettaGrid::map_width() {
  return _grid->width;
}

unsigned int MettaGrid::map_height() {
  return _grid->height;
}

py::list MettaGrid::grid_features() {
  return py::cast(_grid_features);
}

unsigned int MettaGrid::num_agents() {
  return _agents.size();
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  return _episode_rewards;
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
                                      py::arg("dtype") = py::module_::import("numpy").attr("int64"));
}

py::object MettaGrid::observation_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  return spaces.attr("Box")(0,
                            255,
                            py::make_tuple(_obs_height, _obs_width, _grid_features.size()),
                            py::arg("dtype") = py::module_::import("numpy").attr("uint8"));
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

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<py::dict, py::array>())
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
      .def("current_timestep", &MettaGrid::current_timestep)
      .def("map_width", &MettaGrid::map_width)
      .def("map_height", &MettaGrid::map_height)
      .def("grid_features", &MettaGrid::grid_features)
      .def("num_agents", &MettaGrid::num_agents)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def_property_readonly("action_space", &MettaGrid::action_space)
      .def_property_readonly("observation_space", &MettaGrid::observation_space)
      .def("action_success", &MettaGrid::action_success)
      .def("max_action_args", &MettaGrid::max_action_args)
      .def("object_type_names", &MettaGrid::object_type_names)
      .def("inventory_item_names", &MettaGrid::inventory_item_names);
}
