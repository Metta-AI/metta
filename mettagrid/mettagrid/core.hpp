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

// Forward declarations for external classes
class Grid;
class GridObject;
class Agent;
class EventManager;
class StatsTracker;
class ActionHandler;
class ObservationEncoder;

// Type definitions
typedef unsigned int GridObjectId;
typedef unsigned int ActionType;
typedef uint8_t ObsType;
typedef std::map<std::string, int> ActionConfig;

// Main MettaGrid class
class MettaGrid {
private:
  // Grid and environment management
  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;
  std::unique_ptr<ObservationEncoder> _obs_encoder;
  std::unique_ptr<StatsTracker> _stats;

  // Action management
  std::vector<std::unique_ptr<ActionHandler>> _action_handlers;
  int _num_action_handlers;
  std::vector<unsigned char> _max_action_args;
  unsigned char _max_action_arg;
  unsigned char _max_action_priority;

  // Agent management
  std::vector<Agent*> _agents;  // Owned by the grid, not this class

  // Group rewards
  std::map<unsigned int, float> _group_reward_pct;
  std::map<unsigned int, unsigned int> _group_sizes;
  std::vector<double> _group_rewards;

  // Timestep tracking
  unsigned int _current_timestep;
  unsigned int _max_timestep;

  // Observation dimensions
  unsigned short _obs_width;
  unsigned short _obs_height;
  std::vector<std::string> _grid_features;

  // Buffer management - now owned by this class
  std::vector<ObsType> _observations;
  std::vector<char> _terminals;
  std::vector<char> _truncations;
  std::vector<float> _rewards;
  std::vector<float> _episode_rewards;

  // Support for reward decay
  bool _enable_reward_decay;
  float _reward_multiplier;
  float _reward_decay_factor;

  // Action tracking
  bool _track_last_action;
  unsigned char _last_action_obs_idx;
  unsigned char _last_action_arg_obs_idx;
  std::vector<char> _action_success;

public:
  // Constructor - note the changes to take ownership of the grid
  MettaGrid(unsigned int map_width,
            unsigned int map_height,
            unsigned int num_agents,
            unsigned int max_timestep,
            unsigned short obs_width,
            unsigned short obs_height);

  // Destructor - simplified with smart pointers
  ~MettaGrid();

  // Initialization methods
  void init_action_handlers(const std::vector<ActionHandler*>& action_handlers);
  void add_agent(Agent* agent);
  void initialize_from_json(const std::string& map_json, const std::string& config_json);

  // Core game loop methods
  void step(int** actions);
  void reset();

  // Observation methods
  void compute_observation(unsigned short observer_r,
                           unsigned short observer_c,
                           unsigned short obs_width,
                           unsigned short obs_height,
                           ObsType* observation);
  void compute_observations(int** actions);
  void observe(GridObjectId observer_id, unsigned short obs_width, unsigned short obs_height, ObsType* observation);
  void observe_at(unsigned short row,
                  unsigned short col,
                  unsigned short obs_width,
                  unsigned short obs_height,
                  ObsType* observation);

  // Observation utilities
  void observation_at(ObsType* flat_buffer,
                      unsigned int obs_width,
                      unsigned int obs_height,
                      unsigned int feature_size,
                      unsigned int r,
                      unsigned int c,
                      ObsType* output);
  void set_observation_at(ObsType* flat_buffer,
                          unsigned int obs_width,
                          unsigned int obs_height,
                          unsigned int feature_size,
                          unsigned int r,
                          unsigned int c,
                          const ObsType* values);

  // Reward management
  void enable_reward_decay(int decay_time_steps = -1);
  void disable_reward_decay();
  void compute_group_rewards(float* rewards);
  void set_group_reward_pct(unsigned int group_id, float pct);
  void set_group_size(unsigned int group_id, unsigned int size);

  // Getters for buffer access - now return references to internal vectors
  const std::vector<ObsType>& get_observations() const {
    return _observations;
  }
  const std::vector<char>& get_terminals() const {
    return _terminals;
  }
  const std::vector<char>& get_truncations() const {
    return _truncations;
  }
  const std::vector<float>& get_rewards() const {
    return _rewards;
  }
  const std::vector<float>& get_episode_rewards() const {
    return _episode_rewards;
  }
  const std::vector<double>& get_group_rewards() const {
    return _group_rewards;
  }

  // Status and environment information getters
  unsigned int current_timestep() const {
    return _current_timestep;
  }
  unsigned int map_width() const;
  unsigned int map_height() const;
  std::vector<std::string> grid_features() const {
    return _grid_features;
  }
  unsigned int num_agents() const {
    return _agents.size();
  }
  std::vector<char> action_success() const {
    return _action_success;
  }
  std::vector<unsigned char> max_action_args() const {
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
  std::string render_ascii() const;

private:
  // Helper methods for initialization
  void setup_action_handlers(const nlohmann::json& config);
  void parse_grid_object(const std::string& object_type, int row, int col, const nlohmann::json& config);
  void parse_agent(const std::string& group_name,
                   int row,
                   int col,
                   const nlohmann::json& agent_config,
                   const nlohmann::json& group_config);

  // Initialize internal buffers based on number of agents and observation dimensions
  void initialize_buffers(unsigned int num_agents);
};

#endif  // CORE_HPP