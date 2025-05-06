#ifndef CORE_HPP
#define CORE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
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

// Main MettaGrid class
class MettaGrid {
private:
  std::map<unsigned int, float> _group_reward_pct;
  std::map<unsigned int, unsigned int> _group_sizes;
  double* _group_rewards;
  Grid* _grid;
  EventManager* _event_manager;  // Ensure this is a pointer
  unsigned int _current_timestep;
  unsigned int _max_timestep;

  std::vector<ActionHandler*> _action_handlers;
  int _num_action_handlers;
  std::vector<unsigned char> _max_action_args;
  unsigned char _max_action_arg;
  unsigned char _max_action_priority;

  ObservationEncoder* _obs_encoder;  // Ensure this is a pointer
  StatsTracker* _stats;

  unsigned short _obs_width;
  unsigned short _obs_height;

  std::vector<Agent*> _agents;

  ObsType* _observations;
  char* _terminals;
  char* _truncations;
  float* _rewards;
  float* _episode_rewards;

  // Support for reward decay IIR in _step()
  bool _enable_reward_decay;
  float _reward_multiplier;
  float _reward_decay_factor;

  std::vector<std::string> _grid_features;

  bool _track_last_action;
  unsigned char _last_action_obs_idx;
  unsigned char _last_action_arg_obs_idx;
  std::vector<bool> _action_success;

  unsigned int _rewards_size;
  unsigned int _group_rewards_size;
  unsigned int _episode_rewards_size;
  unsigned int _truncations_size;
  unsigned int _terminals_size;
  unsigned int _observations_size;

public:
  // Constructor declaration
  MettaGrid(Grid* grid,
            unsigned int num_agents,
            unsigned int max_timestep,
            unsigned short obs_width,
            unsigned short obs_height);

  // Destructor
  ~MettaGrid();

  // Core methods
  void init_action_handlers(std::vector<ActionHandler*> action_handlers);
  void add_agent(Agent* agent);
  void compute_observation(unsigned short observer_r,
                           unsigned short observer_c,
                           unsigned short obs_width,
                           unsigned short obs_height,
                           ObsType* observation);
  void compute_observations(int** actions);
  void step(int** actions);

  // Getters and utility methods
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

  // Reward decay methods
  void enable_reward_decay(int decay_time_steps = -1);
  void disable_reward_decay();

  // Observation methods
  void observe(GridObjectId observer_id, unsigned short obs_width, unsigned short obs_height, ObsType* observation);
  void observe_at(unsigned short row,
                  unsigned short col,
                  unsigned short obs_width,
                  unsigned short obs_height,
                  ObsType* observation);

  // Accessors
  float* get_episode_rewards() const {
    return _episode_rewards;
  }
  std::vector<bool> action_success() const {
    return _action_success;
  }
  std::vector<unsigned char> max_action_args() const {
    return _max_action_args;
  }

  void set_buffers(ObsType* observations,
                   char* terminals,
                   char* truncations,
                   float* rewards,
                   float* episode_rewards,
                   unsigned int num_agents);

  // Group rewards handling
  void init_group_rewards(double* group_rewards, unsigned int size);
  void set_group_reward_pct(unsigned int group_id, float pct);
  void set_group_size(unsigned int group_id, unsigned int size);
  void compute_group_rewards(float* rewards);

  // StatsTracker accessor
  StatsTracker* stats() const {
    return _stats;
  }
  void set_stats(StatsTracker* s) {
    _stats = s;
  }

  void init_event_manager(EventManager* event_manager);
  EventManager* get_event_manager() {
    return _event_manager;
  }

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
};

#endif  // CORE_HPP