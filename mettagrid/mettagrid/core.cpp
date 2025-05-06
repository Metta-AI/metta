#include "core.hpp"

// Include necessary headers
#include "action_handler.hpp"
#include "event.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "observation_encoder.hpp"
#include "stats_tracker.hpp"

MettaGrid::MettaGrid(Grid* grid,
                     unsigned int num_agents,
                     unsigned int max_timestep,
                     unsigned short obs_width,
                     unsigned short obs_height) {
  _grid = grid;
  _max_timestep = max_timestep;
  _current_timestep = 0;
  _obs_width = obs_width;
  _obs_height = obs_height;
  _stats = nullptr;  // Initialize to null

  // Initialize event manager and observation encoder
  _event_manager = new EventManager();
  _obs_encoder = new ObservationEncoder();

  // Initialize arrays for agents
  _action_success.resize(num_agents);

  // Set up reward decay
  _enable_reward_decay = false;
  _reward_multiplier = 1.0;
  _reward_decay_factor = 3.0 / (_max_timestep > 0 ? _max_timestep : 100);

  // Grid features will be initialized by the observer
  _grid_features = _obs_encoder->feature_names();

  // Initialize buffer sizes to 0
  _rewards_size = 0;
  _episode_rewards_size = 0;
  _truncations_size = 0;
  _terminals_size = 0;
  _observations_size = 0;

  // Initialize pointers to null
  _observations = nullptr;
  _terminals = nullptr;
  _truncations = nullptr;
  _rewards = nullptr;
  _episode_rewards = nullptr;
  _group_rewards = nullptr;
  _group_rewards_size = 0;
}

// Destructor
MettaGrid::~MettaGrid() {
  // Note: we don't delete _grid here because it's passed in by the owner

  // Clean up action handlers
  for (auto handler : _action_handlers) {
    delete handler;
  }

  // Clean up event manager and observation encoder
  delete _event_manager;
  delete _obs_encoder;
}

// Initialize action handlers
void MettaGrid::init_action_handlers(std::vector<ActionHandler*> action_handlers) {
  _action_handlers = action_handlers;
  _num_action_handlers = action_handlers.size();
  _max_action_priority = 0;
  _max_action_arg = 0;
  _max_action_args.resize(action_handlers.size());

  for (unsigned int i = 0; i < action_handlers.size(); i++) {
    ActionHandler* handler = action_handlers[i];
    handler->init(_grid);
    unsigned char max_arg = handler->max_arg();
    _max_action_args[i] = max_arg;
    _max_action_arg = std::max(_max_action_arg, max_arg);
    _max_action_priority = std::max(_max_action_priority, handler->priority);
  }
}

// Add an agent to the game
void MettaGrid::add_agent(Agent* agent) {
  agent->init(&_rewards[_agents.size()]);
  _agents.push_back(agent);
}

// Get observation values at coordinates (r,c)
void MettaGrid::observation_at(ObsType* flat_buffer,
                               unsigned int obs_width,
                               unsigned int obs_height,
                               unsigned int feature_size,
                               unsigned int r,
                               unsigned int c,
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
                                   unsigned int obs_width,
                                   unsigned int obs_height,
                                   unsigned int feature_size,
                                   unsigned int r,
                                   unsigned int c,
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

void MettaGrid::compute_observation(unsigned observer_r,
                                    unsigned int observer_c,
                                    unsigned short obs_width,
                                    unsigned short obs_height,
                                    ObsType* observation) {
  unsigned short obs_width_r = obs_width >> 1;
  unsigned short obs_height_r = obs_height >> 1;
  std::vector<ObsType> temp_obs(_grid_features.size(), 0);

  // Clear the observation buffer
  memset(observation, 0, obs_width * obs_height * _grid_features.size() * sizeof(ObsType));

  unsigned int r_start = std::max(observer_r, (unsigned int)obs_height_r) - obs_height_r;
  unsigned int c_start = std::max(observer_c, (unsigned int)obs_width_r) - obs_width_r;

  for (unsigned int r = r_start; r <= observer_r + obs_height_r; r++) {
    if (r < 0 || r >= _grid->height) continue;

    for (unsigned int c = c_start; c <= observer_c + obs_width_r; c++) {
      if (c < 0 || c >= _grid->width) continue;

      for (unsigned int layer = 0; layer < _grid->num_layers; layer++) {
        GridLocation object_loc(r, c, layer);
        GridObject* obj = _grid->object_at(object_loc);
        if (obj == nullptr) continue;

        unsigned int obs_r = object_loc.r + obs_height_r - observer_r;
        unsigned int obs_c = object_loc.c + obs_width_r - observer_c;

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
void MettaGrid::compute_observations(int** actions) {
  for (unsigned int idx = 0; idx < _agents.size(); idx++) {
    Agent* agent = _agents[idx];
    compute_observation(agent->location.r,
                        agent->location.c,
                        _obs_width,
                        _obs_height,
                        // This would need to be adjusted based on how observations are stored
                        _observations + idx * _obs_width * _obs_height * _grid_features.size());
  }
}

// Take a step in the environment
void MettaGrid::step(int** actions) {
  // Reset rewards and observations
  for (unsigned int i = 0; i < _rewards_size; i++) {
    _rewards[i] = 0;
  }
  // Reset observations would be here

  // Clear success flags
  for (unsigned int i = 0; i < _action_success.size(); i++) {
    _action_success[i] = false;
  }

  _current_timestep++;

  _event_manager->process_events(_current_timestep);

  // Process actions by priority
  for (unsigned char p = 0; p <= _max_action_priority; p++) {
    for (unsigned int idx = 0; idx < _agents.size(); idx++) {
      int action = actions[idx][0];
      if (action < 0 || action >= _num_action_handlers) {
        printf("Invalid action: %d\n", action);
        continue;
      }

      ActionArg arg(actions[idx][1]);
      Agent* agent = _agents[idx];
      ActionHandler* handler = _action_handlers[action];

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
    for (unsigned int i = 0; i < _rewards_size; i++) {
      _rewards[i] *= _reward_multiplier;
    }
  }

  // Update episode rewards
  for (unsigned int i = 0; i < _episode_rewards_size; i++) {
    _episode_rewards[i] += _rewards[i];
  }

  // Check for termination
  if (_max_timestep > 0 && _current_timestep >= _max_timestep) {
    for (unsigned int i = 0; i < _truncations_size; i++) {
      _truncations[i] = 1;
    }
  }
}

// Enable reward decay
void MettaGrid::enable_reward_decay(int decay_time_steps) {
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
void MettaGrid::observe(GridObjectId observer_id,
                        unsigned short obs_width,
                        unsigned short obs_height,
                        ObsType* observation) {
  GridObject* observer = _grid->object(observer_id);
  compute_observation(observer->location.r, observer->location.c, obs_width, obs_height, observation);
}

// Observe from a specific location
void MettaGrid::observe_at(unsigned short row,
                           unsigned short col,
                           unsigned short obs_width,
                           unsigned short obs_height,
                           ObsType* observation) {
  compute_observation(row, col, obs_width, obs_height, observation);
}

// Get map width
unsigned int MettaGrid::map_width() const {
  return _grid->width;
}

// Get map height
unsigned int MettaGrid::map_height() const {
  return _grid->height;
}

// Add buffer initialization
void MettaGrid::set_buffers(ObsType* observations,
                            char* terminals,
                            char* truncations,
                            float* rewards,
                            float* episode_rewards,
                            unsigned int num_agents) {
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _episode_rewards = episode_rewards;

  _rewards_size = num_agents;
  _episode_rewards_size = num_agents;
  _truncations_size = num_agents;
  _terminals_size = num_agents;

  // Initialize group rewards - we'll set proper size later
  _group_rewards = nullptr;
  _group_rewards_size = 0;

  // Re-initialize agents with new reward pointers
  for (unsigned int i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards[i]);
  }
}

void MettaGrid::init_group_rewards(double* group_rewards, unsigned int size) {
  _group_rewards = group_rewards;
  _group_rewards_size = size;
}

// Add group reward computation
void MettaGrid::compute_group_rewards(float* rewards) {
  // Initialize group rewards to 0
  for (unsigned int i = 0; i < _group_rewards_size; i++) {
    _group_rewards[i] = 0;
  }

  bool share_rewards = false;

  // First pass: collect group rewards
  for (unsigned int agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (rewards[agent_idx] != 0) {
      share_rewards = true;
      Agent* agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      float group_reward = rewards[agent_idx] * _group_reward_pct[group_id];
      rewards[agent_idx] -= group_reward;
      _group_rewards[group_id] += group_reward / _group_sizes[group_id];
    }
  }

  // Second pass: distribute group rewards
  if (share_rewards) {
    for (unsigned int agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      Agent* agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      float group_reward = _group_rewards[group_id];
      rewards[agent_idx] += group_reward;
    }
  }
}

void MettaGrid::set_group_reward_pct(unsigned int group_id, float pct) {
  _group_reward_pct[group_id] = pct;
}

void MettaGrid::set_group_size(unsigned int group_id, unsigned int size) {
  _group_sizes[group_id] = size;
}

void MettaGrid::init_event_manager(EventManager* event_manager) {
  if (_event_manager) {
    delete _event_manager;
  }
  _event_manager = event_manager;
}