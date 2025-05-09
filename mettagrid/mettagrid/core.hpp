#ifndef CORE_HPP
#define CORE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "types.hpp"

// Main MettaGrid class
class CppMettaGrid {
private:
  // Grid and environment management
  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;
  std::unique_ptr<StatsTracker> _stats;

  // Action management
  std::vector<std::unique_ptr<ActionHandler>> _action_handlers;
  int32_t _num_action_handlers;
  std::vector<uint8_t> _max_action_args;
  uint8_t _max_action_arg;
  uint8_t _max_action_priority;

  // Agent management
  std::vector<Agent*> _agents;  // Owned by the grid, not this class
  uint32_t _num_agents;

  // Group rewards
  std::map<uint32_t, float> _group_reward_pct;
  std::map<uint32_t, uint32_t> _group_sizes;

  // Timestep tracking
  uint32_t _current_timestep;
  uint32_t _max_timestep;

  // Observation dimensions
  uint16_t _obs_width;
  uint16_t _obs_height;

  std::vector<std::string> _grid_features;

  // Pointers to external buffers - these are required and must be set
  ObsType* _observations;
  int8_t* _terminals;
  int8_t* _truncations;
  float* _rewards;
  float* _episode_rewards;
  float* _group_rewards;

  // Buffer sizes
  size_t _observations_size;
  size_t _terminals_size;
  size_t _truncations_size;
  size_t _rewards_size;
  size_t _episode_rewards_size;
  size_t _group_rewards_size;

  // Support for reward decay
  bool _enable_reward_decay;
  float _reward_multiplier;
  float _reward_decay_factor;

  std::vector<int8_t> _action_success;

public:
  // Constructor - note the changes to take ownership of the grid
  CppMettaGrid(uint32_t map_width,
               uint32_t map_height,
               uint32_t num_agents,
               uint32_t max_timestep,
               uint16_t obs_width,
               uint16_t obs_height);

  // Destructor - simplified with smart pointers
  ~CppMettaGrid();

  // Method to set external buffers - must be called before using the object
  void set_buffers(ObsType* external_observations,
                   int8_t* external_terminals,
                   int8_t* external_truncations,
                   float* external_rewards,
                   float* external_episode_rewards,
                   float* external_group_rewards);

  // Buffer access methods
  ObsType* get_observations() const {
    return _observations;
  }
  int8_t* get_terminals() const {
    return _terminals;
  }
  int8_t* get_truncations() const {
    return _truncations;
  }
  float* get_rewards() const {
    return _rewards;
  }
  float* get_episode_rewards() const {
    return _episode_rewards;
  }
  float* get_group_rewards() const {
    return _group_rewards;
  }

  // Buffer size methods
  size_t get_observations_size() const {
    return _observations_size;
  }
  size_t get_terminals_size() const {
    return _terminals_size;
  }
  size_t get_truncations_size() const {
    return _truncations_size;
  }
  size_t get_rewards_size() const {
    return _rewards_size;
  }
  size_t get_episode_rewards_size() const {
    return _episode_rewards_size;
  }
  size_t get_group_rewards_size() const {
    return _group_rewards_size;
  }

  // Initialization methods
  void init_action_handlers(const std::vector<ActionHandler*>& action_handlers);
  void add_agent(Agent* agent);
  void initialize_from_json(const std::string& map_json, const std::string& config_json);

  // Core game loop methods
  void step(int32_t** actions);
  void reset();

  // Observation methods
  void compute_observation(uint16_t observer_r,
                           uint16_t observer_c,
                           uint16_t obs_width,
                           uint16_t obs_height,
                           ObsType* observation);
  void compute_observations(int32_t** actions);
  void observe(GridObjectId observer_id, uint16_t obs_width, uint16_t obs_height, ObsType* observation);
  void observe_at(uint16_t row,
                  uint16_t col,
                  uint16_t obs_width,
                  uint16_t obs_height,
                  ObsType* observation,
                  uint8_t dummy);

  // Observation utilities
  void observation_at(ObsType* flat_buffer,
                      uint32_t obs_width,
                      uint32_t obs_height,
                      uint32_t feature_size,
                      uint32_t r,
                      uint32_t c,
                      ObsType* output);
  void set_observation_at(ObsType* flat_buffer,
                          uint32_t obs_width,
                          uint32_t obs_height,
                          uint32_t feature_size,
                          uint32_t r,
                          uint32_t c,
                          const ObsType* values);

  // Reward management
  void enable_reward_decay(int32_t decay_time_steps = -1);
  void disable_reward_decay();
  void compute_group_rewards(float* rewards);
  void set_group_reward_pct(uint32_t group_id, float pct);
  void set_group_size(uint32_t group_id, uint32_t size);

  // Status and environment information getters
  uint32_t current_timestep() const {
    return _current_timestep;
  }
  uint32_t map_width() const;
  uint32_t map_height() const;
  std::vector<std::string> grid_features() const {
    return _grid_features;
  }
  uint32_t num_agents() const {
    return _agents.size();
  }
  std::vector<int8_t> action_success() const {
    return _action_success;
  }
  std::vector<uint8_t> max_action_args() const {
    return _max_action_args;
  }
  const std::vector<Agent*>& get_agents() const {
    return _agents;
  }

  // Stats and visualization
  StatsTracker* stats() const {
    return _stats.get();
  }
  EventManager* get_event_manager() {
    return _event_manager.get();
  }
  std::string get_episode_stats_json() const;

  std::vector<std::string> action_names() const;

  std::string get_grid_objects_json() const;

  float get_reward_multiplier() const {
    return _reward_multiplier;
  }

private:
  // Helper methods for initialization
  void setup_action_handlers(const nlohmann::json& config);
  void parse_grid_object(const std::string& object_type, int32_t row, int32_t col, const nlohmann::json& config);
  void parse_agent(const std::string& group_name,
                   int32_t row,
                   int32_t col,
                   const nlohmann::json& agent_config,
                   const nlohmann::json& group_config);
};

#endif  // CORE_HPP