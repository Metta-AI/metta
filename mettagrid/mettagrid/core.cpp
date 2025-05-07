#include "core.hpp"

// Include necessary headers
#include <nlohmann/json.hpp>

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
#include "grid_object.hpp"
#include "objects/constants.hpp"
#include "objects/production_handler.hpp"
#include "observation_encoder.hpp"
#include "stats_tracker.hpp"

MettaGrid::MettaGrid(uint32_t map_width,
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

  // Create the grid - now owned by this class
  _grid = std::make_unique<Grid>(map_width, map_height);

  // Create event manager and observation encoder - now using smart pointers
  _event_manager = std::make_unique<EventManager>();
  _obs_encoder = std::make_unique<ObservationEncoder>();
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
  _reward_multiplier = 1.0f;
  _reward_decay_factor = 3.0f / (_max_timestep > 0 ? _max_timestep : 100);

  // Get grid features from the encoder
  _grid_features = _obs_encoder->feature_names();

  // Initialize internal buffers
  initialize_buffers(num_agents);
}

// Destructor is simpler with smart pointers
MettaGrid::~MettaGrid() {
  // Smart pointers will clean up automatically
  // Agents are owned by the grid, not by us directly
}

void MettaGrid::initialize_buffers(uint32_t num_agents) {
  // Calculate buffer sizes
  uint32_t obs_size = num_agents * _obs_width * _obs_height * _grid_features.size();

  // Resize all buffers
  _observations.resize(obs_size, 0);
  _terminals.resize(num_agents, 0);
  _truncations.resize(num_agents, 0);
  _rewards.resize(num_agents, 0);
  _episode_rewards.resize(num_agents, 0);
  _group_rewards.resize(num_agents, 0);  // Default size, will be adjusted later
}

// Initialize action handlers - modified to take ownership through cloning
void MettaGrid::init_action_handlers(const std::vector<ActionHandler*>& action_handlers) {
  _num_action_handlers = action_handlers.size();
  _max_action_args.resize(_num_action_handlers);
  _max_action_priority = 0;
  _max_action_arg = 0;

  // Clear previous action handlers
  _action_handlers.clear();

  // Create and store copies of the action handlers
  for (uint32_t i = 0; i < action_handlers.size(); i++) {
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
void MettaGrid::add_agent(Agent* agent) {
  // Initialize the agent with its reward buffer
  agent->init(&_rewards[_agents.size()]);
  _agents.push_back(agent);
}

// Reset the environment
void MettaGrid::reset() {
  // Reset timestep
  _current_timestep = 0;

  // Reset buffers
  std::fill(_rewards.begin(), _rewards.end(), 0);
  std::fill(_episode_rewards.begin(), _episode_rewards.end(), 0);
  std::fill(_terminals.begin(), _terminals.end(), 0);
  std::fill(_truncations.begin(), _truncations.end(), 0);
  std::fill(_observations.begin(), _observations.end(), 0);

  // Reset action success flags
  std::fill(_action_success.begin(), _action_success.end(), false);

  // Reset reward decay
  _reward_multiplier = 1.0f;

  // Note: This doesn't reset the grid or agents, which would require re-initialization
}

// Get observation values at coordinates (r,c)
void MettaGrid::observation_at(ObsType* flat_buffer,
                               uint32_t obs_width,
                               uint32_t obs_height,
                               uint32_t feature_size,
                               uint32_t r,
                               uint32_t c,
                               ObsType* output) {
  // Check bounds
  if (r >= obs_height || c >= obs_width) {
    return;
  }

  // Calculate offset in the flat buffer
  ObsType* src = flat_buffer + (r * obs_width + c) * feature_size;

  // Copy to output
  memcpy(output, src, feature_size * sizeof(ObsType));
}

// Set observation values at coordinates (r,c)
void MettaGrid::set_observation_at(ObsType* flat_buffer,
                                   uint32_t obs_width,
                                   uint32_t obs_height,
                                   uint32_t feature_size,
                                   uint32_t r,
                                   uint32_t c,
                                   const ObsType* values) {
  // Check bounds
  if (r >= obs_height || c >= obs_width) {
    return;
  }

  // Calculate offset in the flat buffer
  ObsType* dest = flat_buffer + (r * obs_width + c) * feature_size;

  // Copy from values
  memcpy(dest, values, feature_size * sizeof(ObsType));
}

void MettaGrid::compute_observation(uint16_t observer_r,
                                    uint16_t observer_c,
                                    uint16_t obs_width,
                                    uint16_t obs_height,
                                    ObsType* observation) {
  uint16_t obs_width_r = obs_width >> 1;
  uint16_t obs_height_r = obs_height >> 1;
  std::vector<ObsType> temp_obs(_grid_features.size(), 0);

  // Clear the observation buffer
  memset(observation, 0, obs_width * obs_height * _grid_features.size() * sizeof(ObsType));

  uint32_t r_start = std::max<uint32_t>(observer_r, obs_height_r) - obs_height_r;
  uint32_t c_start = std::max<uint32_t>(observer_c, obs_width_r) - obs_width_r;

  for (uint32_t r = r_start; r <= observer_r + obs_height_r; r++) {
    if (r < 0 || r >= _grid->height) continue;

    for (uint32_t c = c_start; c <= observer_c + obs_width_r; c++) {
      if (c < 0 || c >= _grid->width) continue;

      for (uint32_t layer = 0; layer < _grid->num_layers; layer++) {
        GridLocation object_loc(r, c, layer);
        GridObject* obj = _grid->object_at(object_loc);
        if (obj == nullptr) continue;

        uint32_t obs_r = object_loc.r + obs_height_r - observer_r;
        uint32_t obs_c = object_loc.c + obs_width_r - observer_c;

        // Reset temp buffer
        std::fill(temp_obs.begin(), temp_obs.end(), 0);

        // Encode object into temp buffer
        _obs_encoder->encode(obj, temp_obs.data());

        // Set the observation at the correct location
        set_observation_at(observation, obs_width, obs_height, _grid_features.size(), obs_r, obs_c, temp_obs.data());
      }
    }
  }
}

// Compute observations for all agents
void MettaGrid::compute_observations(int32_t** actions) {
  for (uint32_t idx = 0; idx < _agents.size(); idx++) {
    Agent* agent = _agents[idx];
    ObsType* obs_ptr = _observations.data() + idx * _obs_width * _obs_height * _grid_features.size();

    compute_observation(agent->location.r, agent->location.c, _obs_width, _obs_height, obs_ptr);
  }
}

// Take a step in the environment
void MettaGrid::step(int32_t** actions) {
  // Reset rewards
  std::fill(_rewards.begin(), _rewards.end(), 0);

  // Reset success flags
  std::fill(_action_success.begin(), _action_success.end(), false);

  _current_timestep++;

  _event_manager->process_events(_current_timestep);

  // Process actions by priority
  for (uint8_t p = 0; p <= _max_action_priority; p++) {
    for (uint32_t idx = 0; idx < _agents.size(); idx++) {
      int32_t action = actions[idx][0];
      if (action < 0 || action >= _num_action_handlers) {
        printf("Invalid action: %d\n", action);
        continue;
      }

      ActionArg arg(actions[idx][1]);
      Agent* agent = _agents[idx];
      ActionHandler* handler = _action_handlers[action].get();

      if (handler->priority != _max_action_priority - p) {
        continue;
      }
      if (arg > _max_action_args[action]) {
        continue;
      }

      _action_success[idx] = handler->handle_action(idx, agent->id, arg, _current_timestep);
    }
  }

  compute_observations(actions);

  // Apply reward decay if enabled
  if (_enable_reward_decay) {
    _reward_multiplier = std::max(0.1f, _reward_multiplier * (1.0f - _reward_decay_factor));
    for (uint32_t i = 0; i < _rewards.size(); i++) {
      _rewards[i] *= _reward_multiplier;
    }
  }

  // Update episode rewards
  for (uint32_t i = 0; i < _episode_rewards.size(); i++) {
    _episode_rewards[i] += _rewards[i];
  }

  // Check for termination
  if (_max_timestep > 0 && _current_timestep >= _max_timestep) {
    for (uint32_t i = 0; i < _truncations.size(); i++) {
      _truncations[i] = 1;
    }
  }
}

// Enable reward decay
void MettaGrid::enable_reward_decay(int32_t decay_time_steps) {
  _enable_reward_decay = true;
  _reward_multiplier = 1.0f;  // Reset multiplier to initial value

  // Update decay factor if custom time constant provided
  if (decay_time_steps > 0) {
    _reward_decay_factor = 3.0f / decay_time_steps;
  } else {
    _reward_decay_factor = 0.01f;
  }
}

// Disable reward decay
void MettaGrid::disable_reward_decay() {
  _enable_reward_decay = false;
  _reward_multiplier = 1.0f;  // Reset multiplier to initial value
}

// Observe from a specific object's perspective
void MettaGrid::observe(GridObjectId observer_id, uint16_t obs_width, uint16_t obs_height, ObsType* observation) {
  GridObject* observer = _grid->object(observer_id);
  compute_observation(observer->location.r, observer->location.c, obs_width, obs_height, observation);
}

// Observe from a specific location
void MettaGrid::observe_at(uint16_t row,
                           uint16_t col,
                           uint16_t obs_width,
                           uint16_t obs_height,
                           ObsType* observation,
                           uint8_t dummy) {
  compute_observation(row, col, obs_width, obs_height, observation);
}

// Get map width
uint32_t MettaGrid::map_width() const {
  return _grid->width;
}

// Get map height
uint32_t MettaGrid::map_height() const {
  return _grid->height;
}

// Add group reward computation
void MettaGrid::compute_group_rewards(float* rewards) {
  // Initialize group rewards to 0
  std::fill(_group_rewards.begin(), _group_rewards.end(), 0);

  bool share_rewards = false;

  // First pass: collect group rewards
  for (uint32_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (rewards[agent_idx] != 0) {
      share_rewards = true;
      Agent* agent = _agents[agent_idx];
      uint32_t group_id = agent->group;
      float group_reward = rewards[agent_idx] * _group_reward_pct[group_id];
      rewards[agent_idx] -= group_reward;
      _group_rewards[group_id] += group_reward / _group_sizes[group_id];
    }
  }

  // Second pass: distribute group rewards
  if (share_rewards) {
    for (uint32_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      Agent* agent = _agents[agent_idx];
      uint32_t group_id = agent->group;
      float group_reward = _group_rewards[group_id];
      rewards[agent_idx] += group_reward;
    }
  }
}

void MettaGrid::set_group_reward_pct(uint32_t group_id, float pct) {
  _group_reward_pct[group_id] = pct;
}

void MettaGrid::set_group_size(uint32_t group_id, uint32_t size) {
  _group_sizes[group_id] = size;
}

void MettaGrid::initialize_from_json(const std::string& map_json, const std::string& config_json) {
  using json = nlohmann::json;

  // Parse JSON strings
  json map_data = json::parse(map_json);
  json cfg = json::parse(config_json);

  // Initialize group sizes
  std::map<uint32_t, uint32_t> group_sizes;
  for (auto& [group_name, group_info] : cfg["groups"].items()) {
    group_sizes[group_info["id"]] = 0;
  }

  // Update group rewards size
  _group_rewards.resize(cfg["groups"].size(), 0);

  // Process map and create objects
  for (int32_t r = 0; r < map_data.size(); r++) {
    for (int32_t c = 0; c < map_data[r].size(); c++) {
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
}

void MettaGrid::parse_grid_object(const std::string& object_type,
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

void MettaGrid::parse_agent(const std::string& group_name,
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

void MettaGrid::setup_action_handlers(const nlohmann::json& cfg) {
  std::vector<ActionHandler*> temp_handlers;

  // Check each action type and add handlers if enabled
  if (cfg.contains("actions")) {
    auto& actions_cfg = cfg["actions"];

    // PutRecipeItems
    if (actions_cfg.contains("put_items") && actions_cfg["put_items"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["put_items"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new PutRecipeItems(action_config));
    }

    // GetOutput
    if (actions_cfg.contains("get_items") && actions_cfg["get_items"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["get_items"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new GetOutput(action_config));
    }

    // Noop
    if (actions_cfg.contains("noop") && actions_cfg["noop"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["noop"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new Noop(action_config));
    }

    // Move
    if (actions_cfg.contains("move") && actions_cfg["move"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["move"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new Move(action_config));
    }

    // Rotate
    if (actions_cfg.contains("rotate") && actions_cfg["rotate"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["rotate"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new Rotate(action_config));
    }

    // Attack
    if (actions_cfg.contains("attack") && actions_cfg["attack"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["attack"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new Attack(action_config));
      // For AttackNearest, reuse the same config
      temp_handlers.push_back(new AttackNearest(action_config));
    }

    // Swap
    if (actions_cfg.contains("swap") && actions_cfg["swap"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["swap"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new Swap(action_config));
    }

    // ChangeColor
    if (actions_cfg.contains("change_color") && actions_cfg["change_color"]["enabled"]) {
      ActionConfig action_config;
      for (auto& [key, value] : actions_cfg["change_color"].items()) {
        if (value.is_number_integer()) {
          action_config[key] = value;
        }
      }
      temp_handlers.push_back(new ChangeColorAction(action_config));
    }
  }

  // Initialize the action handlers and take ownership
  init_action_handlers(temp_handlers);

  // Clean up temporary handlers that have been cloned and are no longer needed
  for (auto handler : temp_handlers) {
    delete handler;
  }
}

std::string MettaGrid::get_episode_stats_json() const {
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
  for (uint32_t i = 0; i < _agents.size(); i++) {
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

std::string MettaGrid::render_ascii() const {
  // Create an empty grid filled with spaces
  std::vector<std::vector<char>> grid(_grid->height, std::vector<char>(_grid->width, ' '));

  // Iterate through objects and update grid
  for (size_t obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    GridObject* obj = _grid->object(obj_id);
    grid[obj->location.r][obj->location.c] = ObjectTypeAscii[obj->_type_id][0];
  }

  // Convert the 2D grid to a string representation
  std::string result;
  for (const auto& row : grid) {
    for (char c : row) {
      result += c;
    }
    result += '\n';  // Add newline after each row
  }

  return result;
}