#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"

// Forward declarations
struct ActionConfig;
struct GridObjectConfig;

using ObservationCoord = ObservationType;

struct GlobalObsConfig {
  bool episode_completion_pct = true;
  bool last_action = true;
  bool last_reward = true;
  bool resource_rewards = false;
  bool visitation_counts = false;
};

struct GameConfig {
  size_t num_agents;
  unsigned int max_steps;
  bool episode_truncates;
  ObservationCoord obs_width;
  ObservationCoord obs_height;
  std::vector<std::string> inventory_item_names;
  unsigned int num_observation_tokens;
  GlobalObsConfig global_obs;
  std::map<std::string, std::shared_ptr<ActionConfig>> actions;
  std::map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  float resource_loss_prob = 0.0;

  // feature flags
  bool track_movement_metrics = false;
  bool no_agent_interference = false;
  bool recipe_details_obs = false;
  bool allow_diagonals = false;
};

#endif  // CONFIG_HPP_
