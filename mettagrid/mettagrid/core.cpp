#include "core.hpp"

// Include necessary headers
#include <nlohmann/json.hpp>

#include "actions/action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/attack_nearest.hpp"
#include "actions/change_color.hpp"
#include "actions/get_output.hpp"
#include "actions/move.hpp"
#include "actions/noop.hpp"
#include "actions/put_recipe_items.hpp"
#include "actions/rotate.hpp"
#include "actions/swap.hpp"
#include "constants.hpp"
#include "event_handlers.hpp"
#include "event_manager.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"
#include "stats_tracker.hpp"
#include "types.hpp"

// Constructor needs to be updated to use the new feature_names from GridObject
CppMettaGrid::CppMettaGrid(uint32_t map_width,
                           uint32_t map_height,
                           uint32_t num_agents,
                           uint32_t max_timestep,
                           uint16_t obs_width,
                           uint16_t obs_height) {
  // Initialize member variables
  _current_timestep = 0;
  _max_timestep = max_timestep;
  _obs_width = obs_width;
  _obs_height = obs_height;
  _num_agents = num_agents;

  // Create the grid - now owned by this class
  _grid = std::make_unique<Grid>(map_width, map_height);

  // Create event manager and stats tracker - now using smart pointers
  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>();

  // Initialize the event manager
  _event_manager->init(_grid.get(), _stats.get());

  // Add event handlers
  _event_manager->event_handlers.push_back(new ProductionHandler(_event_manager.get()));
  _event_manager->event_handlers.push_back(new CoolDownHandler(_event_manager.get()));

  // Set up action success tracking
  _action_success.resize(num_agents);

  // Set up reward decay
  _enable_reward_decay = false;
  _reward_decay_multiplier = 1.0f;
  _reward_decay_factor = 3.0f / (_max_timestep > 0 ? _max_timestep : 100);

  // Get grid features from GridObject instead of the encoder
  _grid_features = GridObject::get_feature_names();

  // Initialize buffer pointers to NULL - they must be set before use
  _observations = nullptr;
  _terminals = nullptr;
  _truncations = nullptr;
  _rewards = nullptr;

  // Initialize group rewards with default size of 0
  // We'll resize when we know the number of groups
  _group_rewards.clear();

  // Calculate required buffer sizes
  _observations_size = num_agents * _obs_width * _obs_height * GridObject::get_observation_size();
  _terminals_size = num_agents;
  _truncations_size = num_agents;
  _rewards_size = num_agents;

  _episode_rewards.resize(_num_agents, 0.0f);
}

// Destructor is simpler with smart pointers
CppMettaGrid::~CppMettaGrid() {
  // Smart pointers will clean up automatically
  // Agents are owned by the grid, not by us directly
  // Most of our memory is externally managed (buffer pointers)
}

void CppMettaGrid::set_buffers(c_observations_type* external_observations,
                               numpy_bool_t* external_terminals,
                               numpy_bool_t* external_truncations,
                               float* external_rewards) {
  // Check for NULL pointers
  if (external_observations == nullptr) {
    throw std::invalid_argument("External buffers cannot be NULL");
  }

  // Set pointers to external buffers
  _observations = external_observations;
  _terminals = external_terminals;
  _truncations = external_truncations;
  _rewards = external_rewards;
}

void CppMettaGrid::init_action_handlers(const std::vector<ActionHandler*>& action_handlers) {
  _num_action_handlers = action_handlers.size();
  _max_action_args.resize(_num_action_handlers);
  _max_action_priority = 0;
  _max_action_arg = 0;

  // Clear previous action handlers
  _action_handlers.clear();

  // Create and store copies of the action handlers
  for (size_t i = 0; i < action_handlers.size(); i++) {
    ActionHandler* handler = action_handlers[i];

    // Create a deep copy through cloning and store it (with ownership)
    _action_handlers.push_back(std::unique_ptr<ActionHandler>(handler->clone()));
    _action_handlers.back()->init(_grid.get());

    // Update maximums
    uint8_t max_arg = handler->max_arg();
    _max_action_args[i] = max_arg;
    _max_action_arg = std::max(_max_action_arg, max_arg);
    _max_action_priority = std::max(_max_action_priority, handler->priority);
  }
}

// Add an agent to the game
void CppMettaGrid::add_agent(Agent* agent) {
  // Initialize the agent with its reward buffer
  agent->init(&_rewards[_agents.size()]);
  _agents.push_back(agent);
}

// Reset the environment
void CppMettaGrid::reset() {
  if (_observations == nullptr) {
    throw std::runtime_error("External buffers not set. Call set_buffers before reset.");
  }

  // Reset timestep
  _current_timestep = 0;

  // Reset external buffers
  std::fill(_rewards, _rewards + _rewards_size, 0);
  std::fill(_terminals, _terminals + _terminals_size, 0);
  std::fill(_truncations, _truncations + _truncations_size, 0);
  std::fill(_observations, _observations + _observations_size, 0);

  // reset internal buffers
  std::fill(_group_rewards.begin(), _group_rewards.end(), 0.0f);
  std::fill(_episode_rewards.begin(), _episode_rewards.end(), 0.0f);

  // Reset action success flags
  std::fill(_action_success.begin(), _action_success.end(), false);

  // Reset reward decay
  _reward_decay_multiplier = 1.0f;
}

// Get observation values at coordinates (r,c)
void CppMettaGrid::observation_at(c_observations_type* flat_buffer,
                                  uint32_t obs_width,
                                  uint32_t obs_height,
                                  uint32_t feature_size,
                                  uint32_t r,
                                  uint32_t c,
                                  c_observations_type* output) {
  // Check bounds
  if (r >= obs_height || c >= obs_width) {
    return;
  }

  // Calculate offset in the flat buffer
  c_observations_type* src = flat_buffer + (r * obs_width + c) * feature_size;

  // Copy to output
  memcpy(output, src, feature_size * sizeof(c_observations_type));
}

// Set observation values at coordinates (r,c)
void CppMettaGrid::set_observation_at(c_observations_type* flat_buffer,
                                      uint32_t obs_width,
                                      uint32_t obs_height,
                                      uint32_t feature_size,
                                      uint32_t r,
                                      uint32_t c,
                                      const c_observations_type* values) {
  // Check bounds
  if (r >= obs_height || c >= obs_width) {
    return;
  }

  // Calculate offset in the flat buffer
  c_observations_type* dest = flat_buffer + (r * obs_width + c) * feature_size;

  // Copy from values
  memcpy(dest, values, feature_size * sizeof(c_observations_type));
}

void CppMettaGrid::compute_observation(uint16_t observer_r,
                                       uint16_t observer_c,
                                       uint16_t obs_width,
                                       uint16_t obs_height,
                                       c_observations_type* observation) {
  uint16_t obs_width_r = obs_width >> 1;
  uint16_t obs_height_r = obs_height >> 1;

  // Get observation size from GridObject's static method
  std::vector<c_observations_type> temp_obs(GridObject::get_observation_size(), 0);

  // Clear the observation buffer
  memset(observation, 0, obs_width * obs_height * GridObject::get_observation_size() * sizeof(c_observations_type));

  uint32_t r_start = std::max<uint32_t>(observer_r, obs_height_r) - obs_height_r;
  uint32_t c_start = std::max<uint32_t>(observer_c, obs_width_r) - obs_width_r;

  for (uint32_t r = r_start; r <= observer_r + obs_height_r; r++) {
    if (r >= _grid->height) continue;  // uint32_t is always >= 0

    for (uint32_t c = c_start; c <= observer_c + obs_width_r; c++) {
      if (c >= _grid->width) continue;  // uint32_t is always >= 0
      for (uint32_t layer = 0; layer < _grid->num_layers; layer++) {
        GridLocation object_loc(r, c, layer);
        GridObject* obj = _grid->object_at(object_loc);
        if (obj == nullptr) continue;

        uint32_t obs_r = object_loc.r + obs_height_r - observer_r;
        uint32_t obs_c = object_loc.c + obs_width_r - observer_c;

        // Reset temp buffer
        std::fill(temp_obs.begin(), temp_obs.end(), 0);

        // Updated: Use the object's own obs method instead of the encoder
        obj->obs(temp_obs.data());

        // Set the observation at the correct location
        set_observation_at(
            observation, obs_width, obs_height, GridObject::get_observation_size(), obs_r, obs_c, temp_obs.data());
      }
    }
  }
}

void CppMettaGrid::compute_observations(c_actions_type* flat_actions) {
  for (size_t idx = 0; idx < _agents.size(); idx++) {
    Agent* agent = _agents[idx];
    c_observations_type* obs = _observations + idx * _obs_width * _obs_height * _grid_features.size();
    compute_observation(agent->location.r, agent->location.c, _obs_width, _obs_height, obs);
  }
}

// Take a step in the environment
void CppMettaGrid::step(c_actions_type* flat_actions) {
  if (flat_actions == nullptr) {
    throw std::runtime_error("Null actions array passed to step()");
  }

  // Check if buffers are set
  if (_observations == nullptr || _rewards == nullptr || _terminals == nullptr || _truncations == nullptr) {
    throw std::runtime_error("External buffers not set. Call set_buffers before step()");
  }

  // Check if action handlers are initialized
  if (_action_handlers.empty()) {
    throw std::runtime_error("Action handlers not initialized. Call init_action_handlers before step()");
  }

  // Reset rewards
  std::fill(_rewards, _rewards + _rewards_size, 0);

  // Reset success flags
  std::fill(_action_success.begin(), _action_success.end(), false);

  _current_timestep++;

  _event_manager->process_events(_current_timestep);

  // Process actions by priority
  for (uint8_t p = 0; p <= _max_action_priority; p++) {
    for (size_t idx = 0; idx < _agents.size(); idx++) {
      c_actions_type action = flat_actions[idx * 2];   // First element is action type
      c_actions_type arg = flat_actions[idx * 2 + 1];  // Second element is action argument

      assert(action >= 0 && "Action cannot be negative");
      assert(action < static_cast<int32_t>(_num_action_handlers) && "Action exceeds available handlers");

      Agent* agent = _agents[idx];
      ActionHandler* handler = _action_handlers[static_cast<size_t>(action)].get();

      assert(handler != nullptr && "Action handler is null");
      assert(handler->priority == _max_action_priority - p || "Action handled in wrong priority phase");
      assert(arg <= _max_action_args[static_cast<size_t>(action)] && "Action argument exceeds maximum allowed");
      assert(_current_timestep > 0 && "Current timestep must be positive");
      assert(agent != nullptr && "Agent is null");
      assert(agent->id > 0 && "Agent ID must be positive");
      assert(agent->id < _grid->objects.size() && "Agent ID exceeds grid object count");

      numpy_bool_t result = handler->handle_action(idx, agent->id, arg, _current_timestep);

      // TODO: this line is causing a segfault
      _action_success[idx] = result;
    }
  }

  compute_observations(flat_actions);

  // Apply reward decay if enabled
  if (_enable_reward_decay) {
    _reward_decay_multiplier = std::max(0.1f, _reward_decay_multiplier * (1.0f - _reward_decay_factor));
    for (size_t i = 0; i < _rewards_size; i++) {
      _rewards[i] *= _reward_decay_multiplier;
    }
  }

  // Update episode rewards
  for (size_t i = 0; i < _episode_rewards.size(); i++) {
    _episode_rewards[i] += _rewards[i];
  }

  // Check for termination
  if (_max_timestep > 0 && _current_timestep >= _max_timestep) {
    for (size_t i = 0; i < _truncations_size; i++) {
      _truncations[i] = 1;
    }
  }
}

// Enable reward decay
void CppMettaGrid::enable_reward_decay(int32_t decay_time_steps) {
  _enable_reward_decay = true;
  _reward_decay_multiplier = 1.0f;  // Reset multiplier to initial value

  // Update decay factor if custom time constant provided
  if (decay_time_steps > 0) {
    _reward_decay_factor = 3.0f / decay_time_steps;
  } else {
    _reward_decay_factor = 0.01f;
  }
}

// Disable reward decay
void CppMettaGrid::disable_reward_decay() {
  _enable_reward_decay = false;
  _reward_decay_multiplier = 1.0f;  // Reset multiplier to initial value
}

// Observe from a specific object's perspective
void CppMettaGrid::observe(GridObjectId observer_id,
                           uint16_t obs_width,
                           uint16_t obs_height,
                           c_observations_type* observation) {
  GridObject* observer = _grid->object(observer_id);
  compute_observation(observer->location.r, observer->location.c, obs_width, obs_height, observation);
}

// Observe from a specific location
void CppMettaGrid::observe_at(uint16_t row,
                              uint16_t col,
                              uint16_t obs_width,
                              uint16_t obs_height,
                              c_observations_type* observation,
                              uint8_t dummy) {
  compute_observation(row, col, obs_width, obs_height, observation);
}

// Get map width
uint32_t CppMettaGrid::map_width() const {
  return _grid->width;
}

// Get map height
uint32_t CppMettaGrid::map_height() const {
  return _grid->height;
}

void CppMettaGrid::compute_group_rewards(float* rewards) {
  // Calculate the required size for the accumulator
  size_t max_group_id = 0;
  for (const auto& agent : _agents) {
    max_group_id = std::max(max_group_id, static_cast<size_t>(agent->group));
  }

  // Create a local vector for accumulation (with size max_group_id + 1)
  std::vector<float> group_rewards_accumulator(max_group_id + 1, 0.0f);

  bool share_rewards = false;

  // First pass: collect group rewards
  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (rewards[agent_idx] != 0) {
      share_rewards = true;
      Agent* agent = _agents[agent_idx];
      uint32_t group_id = agent->group;

      // Skip if group_id is out of bounds
      if (group_id >= group_rewards_accumulator.size()) {
        continue;
      }

      // Get group reward percentage (defaults to 0 if not set)
      float group_pct = 0.0f;
      auto pct_it = _group_reward_pct.find(group_id);
      if (pct_it != _group_reward_pct.end()) {
        group_pct = pct_it->second;
      }

      float group_reward = rewards[agent_idx] * group_pct;
      rewards[agent_idx] -= group_reward;

      // Get group size (defaults to 1 if not set)
      uint32_t group_size = 1;
      auto size_it = _group_sizes.find(group_id);
      if (size_it != _group_sizes.end() && size_it->second > 0) {
        group_size = size_it->second;
      }

      group_rewards_accumulator[group_id] += group_reward / group_size;
    }
  }

  // Second pass: distribute group rewards
  if (share_rewards) {
    // Make sure _group_rewards is properly sized
    if (_group_rewards.size() < group_rewards_accumulator.size()) {
      std::cerr << "Warning: _group_rewards buffer may be too small" << std::endl;
    }

    // Zero out the group rewards buffer
    size_t copy_size = std::min(_group_rewards.size(), group_rewards_accumulator.size());
    std::fill(_group_rewards.begin(), _group_rewards.end(), 0.0f);

    // Copy accumulated rewards to the output buffer up to the smaller size
    for (size_t i = 0; i < copy_size; i++) {
      _group_rewards[i] = group_rewards_accumulator[i];
    }

    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      Agent* agent = _agents[agent_idx];
      uint32_t group_id = agent->group;

      // Skip if group_id is out of bounds
      if (group_id >= group_rewards_accumulator.size()) {
        continue;
      }

      float group_reward = group_rewards_accumulator[group_id];
      rewards[agent_idx] += group_reward;
    }
  }

  // Vector will be automatically destroyed when function exits
}

void CppMettaGrid::set_group_reward_pct(uint32_t group_id, float pct) {
  _group_reward_pct[group_id] = pct;
}

void CppMettaGrid::set_group_size(uint32_t group_id, uint32_t size) {
  _group_sizes[group_id] = size;
}

void CppMettaGrid::initialize_from_json(const std::string& map_json, const std::string& config_json) {
  using json = nlohmann::json;

  // Parse JSON strings
  json map_data = json::parse(map_json);
  json cfg = json::parse(config_json);

  // Initialize group sizes
  std::map<uint32_t, uint32_t> group_sizes;
  for (auto& [group_name, group_info] : cfg["groups"].items()) {
    group_sizes[group_info["id"]] = 0;
  }

  // Get the number of groups and resize vectors
  size_t num_groups = cfg["groups"].size();
  _group_rewards.resize(num_groups, 0.0f);

  // Process map and create objects
  for (size_t r = 0; r < map_data.size(); r++) {
    for (size_t c = 0; c < map_data[r].size(); c++) {
      std::string cell_type = map_data[r][c];

      // Wall
      if (cell_type == "wall") {
        parse_grid_object("wall", r, c, cfg["objects"]["wall"]);
      }
      // Block
      else if (cell_type == "block") {
        parse_grid_object("block", r, c, cfg["objects"]["block"]);
      }
      // Mine
      else if (cell_type.substr(0, 4) == "mine") {
        std::string mine_type = cell_type;
        if (mine_type.find(".") == std::string::npos) {
          mine_type = "mine.red";
        }
        parse_grid_object(mine_type, r, c, cfg["objects"][mine_type]);
      }
      // Generator
      else if (cell_type.substr(0, 9) == "generator") {
        std::string generator_type = cell_type;
        if (generator_type.find(".") == std::string::npos) {
          generator_type = "generator.red";
        }
        parse_grid_object(generator_type, r, c, cfg["objects"][generator_type]);
      }
      // Altar
      else if (cell_type == "altar") {
        parse_grid_object("altar", r, c, cfg["objects"]["altar"]);
      }
      // Armory
      else if (cell_type == "armory") {
        parse_grid_object("armory", r, c, cfg["objects"]["armory"]);
      }
      // Lasery
      else if (cell_type == "lasery") {
        parse_grid_object("lasery", r, c, cfg["objects"]["lasery"]);
      }
      // Lab
      else if (cell_type == "lab") {
        parse_grid_object("lab", r, c, cfg["objects"]["lab"]);
      }
      // Factory
      else if (cell_type == "factory") {
        parse_grid_object("factory", r, c, cfg["objects"]["factory"]);
      }
      // Temple
      else if (cell_type == "temple") {
        parse_grid_object("temple", r, c, cfg["objects"]["temple"]);
      }
      // Agent
      else if (cell_type.substr(0, 6) == "agent.") {
        size_t pos = cell_type.find(".");
        std::string group_name = cell_type.substr(pos + 1);
        parse_agent(group_name, r, c, cfg["agent"], cfg["groups"][group_name]);

        // Update group size
        uint32_t group_id = cfg["groups"][group_name]["id"];
        group_sizes[group_id]++;
        set_group_size(group_id, group_sizes[group_id]);
      }
    }
  }

  // Initialize group reward percentages
  for (auto& [group_name, group_info] : cfg["groups"].items()) {
    uint32_t group_id = group_info["id"];
    float pct = 0;
    if (group_info.contains("group_reward_pct")) {
      pct = group_info["group_reward_pct"];
    }
    set_group_reward_pct(group_id, pct);
  }

  // Set up action handlers from the config
  setup_action_handlers(cfg);

  size_t obs_size = _num_agents * _obs_width * _obs_height * GridObject::get_observation_size();
  _observations_size = obs_size;
}

void CppMettaGrid::parse_grid_object(const std::string& object_type,
                                     int32_t row,
                                     int32_t col,
                                     const nlohmann::json& config) {
  // Convert JSON config to ObjectConfig
  ObjectConfig obj_config;
  for (auto& [key, value] : config.items()) {
    if (value.is_number_integer()) {
      obj_config[key] = value;
    }
  }

  // Create appropriate object based on type
  GridObject* obj = nullptr;

  if (object_type == "wall" || object_type == "block") {
    obj = new Wall(row, col, obj_config);
  } else if (object_type.substr(0, 4) == "mine") {
    obj = new Converter(row, col, obj_config, ObjectType::MineT);
  } else if (object_type.substr(0, 9) == "generator") {
    obj = new Converter(row, col, obj_config, ObjectType::GeneratorT);
  } else if (object_type == "altar") {
    obj = new Converter(row, col, obj_config, ObjectType::AltarT);
  } else if (object_type == "armory") {
    obj = new Converter(row, col, obj_config, ObjectType::ArmoryT);
  } else if (object_type == "lasery") {
    obj = new Converter(row, col, obj_config, ObjectType::LaseryT);
  } else if (object_type == "lab") {
    obj = new Converter(row, col, obj_config, ObjectType::LabT);
  } else if (object_type == "factory") {
    obj = new Converter(row, col, obj_config, ObjectType::FactoryT);
  } else if (object_type == "temple") {
    obj = new Converter(row, col, obj_config, ObjectType::TempleT);
  }

  // Add object to grid
  if (obj != nullptr) {
    _grid->add_object(obj);

    // Update stats
    std::string stat = "objects." + object_type;
    _stats->incr(stat);

    // Set event manager for converters
    Converter* converter = dynamic_cast<Converter*>(obj);
    if (converter != nullptr) {
      converter->set_event_manager(_event_manager.get());
    }
  }
}

void CppMettaGrid::parse_agent(const std::string& group_name,
                               int32_t row,
                               int32_t col,
                               const nlohmann::json& agent_config,
                               const nlohmann::json& group_config) {
  // Merge agent and group configs
  nlohmann::json merged_config = agent_config;
  if (group_config.contains("props")) {
    for (auto& [key, value] : group_config["props"].items()) {
      merged_config[key] = value;
    }
  }

  // Extract rewards
  nlohmann::json rewards = merged_config.value("rewards", nlohmann::json::object());
  if (merged_config.contains("rewards")) {
    merged_config.erase("rewards");
  }

  // Convert to ObjectConfig for Agent creation
  ObjectConfig obj_config;
  for (auto& [key, value] : merged_config.items()) {
    if (value.is_number_integer() || value.is_number_float()) {
      obj_config[key] = static_cast<int32_t>(value.get<double>());
    }
  }

  // Build rewards map
  std::map<std::string, float> cpp_rewards;
  for (const auto& item : InventoryItemNames) {
    // Default value is 0
    cpp_rewards[item] = 0;
    if (rewards.contains(item)) {
      // Convert to float explicitly
      cpp_rewards[item] = static_cast<float>(rewards[item].get<double>());
    }

    // Set max values
    std::string max_key = item + "_max";
    // Default max is 1000
    cpp_rewards[max_key] = 1000.0f;
    if (rewards.contains(max_key)) {
      cpp_rewards[max_key] = static_cast<float>(rewards[max_key].get<double>());
    }
  }

  // Create agent
  uint32_t group_id = group_config["id"];
  Agent* agent = new Agent(row, col, group_name, group_id, obj_config, cpp_rewards);
  agent->agent_id = _agents.size();

  // Add to grid and agent list
  _grid->add_object(agent);
  add_agent(agent);
}

// Modified register_action function
template <typename HandlerType>
void register_action(std::vector<ActionHandler*>& handlers,
                     const nlohmann::json& cfg,
                     const std::string& config_name,
                     ActionType action_type) {
  static_assert(std::is_base_of<ActionHandler, HandlerType>::value, "HandlerType must be derived from ActionHandler");

  if (cfg.contains("actions") && cfg["actions"].contains(config_name) && cfg["actions"][config_name]["enabled"]) {
    // Extract config
    ActionConfig action_config;
    for (auto& [key, value] : cfg["actions"][config_name].items()) {
      if (value.is_number_integer()) {
        action_config[key] = static_cast<int>(value);
      }
    }
    // Create handler
    handlers[action_type] = new HandlerType(action_config);
  }
}

void CppMettaGrid::setup_action_handlers(const nlohmann::json& cfg) {
  std::vector<ActionHandler*> temp_handlers;
  temp_handlers.resize(ActionType::ActionCount, nullptr);

  // Register each action handler - use the HANDLER CLASSES, not enum values
  register_action<Actions::PutRecipeItems>(temp_handlers, cfg, "put_items", ActionType::PutRecipeItems);
  register_action<Actions::GetOutput>(temp_handlers, cfg, "get_items", ActionType::GetOutput);
  register_action<Actions::Noop>(temp_handlers, cfg, "noop", ActionType::Noop);
  register_action<Actions::Move>(temp_handlers, cfg, "move", ActionType::Move);
  register_action<Actions::Rotate>(temp_handlers, cfg, "rotate", ActionType::Rotate);
  register_action<Actions::Attack>(temp_handlers, cfg, "attack", ActionType::Attack);
  register_action<Actions::AttackNearest>(temp_handlers, cfg, "attack", ActionType::AttackNearest);
  register_action<Actions::Swap>(temp_handlers, cfg, "swap", ActionType::Swap);
  register_action<Actions::ChangeColor>(temp_handlers, cfg, "change_color", ActionType::ChangeColor);

  // Rest of the code remains the same
  std::vector<ActionHandler*> final_handlers;
  for (auto handler : temp_handlers) {
    if (handler != nullptr) {
      final_handlers.push_back(handler);
    }
  }

  init_action_handlers(final_handlers);

  for (auto handler : final_handlers) {
    delete handler;
  }
}

std::vector<std::string> CppMettaGrid::action_names() const {
  std::vector<std::string> names;
  for (const auto& handler : _action_handlers) {
    names.push_back(handler->action_name());
  }
  return names;
}

std::string CppMettaGrid::get_episode_stats_json() const {
  nlohmann::json stats_json;

  // Game stats
  nlohmann::json game_stats;
  std::map<std::string, int32_t> cpp_stats = _stats->stats();
  for (const auto& [key, value] : cpp_stats) {
    game_stats[key] = value;
  }
  stats_json["game"] = game_stats;

  // Agent stats
  nlohmann::json agent_stats_array = nlohmann::json::array();
  for (size_t i = 0; i < _agents.size(); i++) {
    Agent* agent = _agents[i];
    nlohmann::json agent_stats;
    std::map<std::string, int32_t> agent_stat_map = agent->stats.stats();

    for (const auto& [key, value] : agent_stat_map) {
      agent_stats[key] = value;
    }

    agent_stats_array.push_back(agent_stats);
  }
  stats_json["agent"] = agent_stats_array;

  return stats_json.dump();
}

std::string CppMettaGrid::get_grid_objects_json() const {
  // Use a std::map to automatically sort keys
  std::map<int, nlohmann::json> sorted_objects;

  for (size_t obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    GridObject* obj = _grid->object(obj_id);
    if (obj == nullptr) continue;

    // Create object entry with basic info
    nlohmann::json obj_json;
    obj_json["id"] = obj_id;

    // Add string type name
    if (obj->_type_id < ObjectTypeNames.size()) {
      obj_json["type_name"] = ObjectTypeNames[obj->_type_id];
      obj_json["type"] = obj->_type_id;
    } else {
      obj_json["type_name"] = "Unknown";
      obj_json["type"] = -1;
    }

    // Add location info
    obj_json["r"] = obj->location.r;
    obj_json["c"] = obj->location.c;
    obj_json["layer"] = obj->location.layer;

    // Updated: Get object features using the object's own obs method
    std::vector<c_observations_type> obj_data(GridObject::get_observation_size(), 0);
    obj->obs(obj_data.data());

    // Add features to JSON, using GridObject::get_feature_names() for mapping
    const auto& feature_names = GridObject::get_feature_names();
    for (size_t i = 0; i < feature_names.size(); i++) {
      const std::string& feature_name = feature_names[i];
      c_observations_type feature_value = obj_data[i];

      // Skip zero values
      if (feature_value == 0) {
        continue;
      }

      // Validate specific features
      bool is_valid = true;
      if (feature_name == "color" && feature_value > 2) {
        is_valid = false;
      } else if (feature_name == "swappable" && feature_value > 1) {
        is_valid = false;
      }

      if (!is_valid) {
        std::cerr << "Warning: Invalid value " << static_cast<int>(feature_value) << " for (" << feature_name
                  << ") on object [" << obj_id << "] of type =" << obj_json["type_name"].get<std::string>()
                  << std::endl;
        continue;
      }

      // Add valid feature to the JSON
      obj_json[feature_name] = static_cast<int>(feature_value);
    }

    // Add to sorted map
    sorted_objects[static_cast<int>(obj_id)] = obj_json;
  }

  // Add agent_id to agent objects
  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    Agent* agent = _agents[agent_idx];
    int agent_id = static_cast<int>(agent->id);
    if (sorted_objects.count(agent_id)) {
      sorted_objects[agent_id]["agent_id"] = agent_idx;
    }
  }

  // Create final JSON object
  nlohmann::json objects_json = nlohmann::json::object();
  for (const auto& [id, obj_json] : sorted_objects) {
    objects_json[std::to_string(id)] = obj_json;
  }

  return objects_json.dump();
}
