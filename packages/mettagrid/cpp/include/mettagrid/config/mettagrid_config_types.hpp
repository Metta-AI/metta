#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_TYPES_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_TYPES_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "systems/clipper_config.hpp"

// Forward declarations
struct ActionConfig;
struct GridObjectConfig;

using ObservationCoord = ObservationType;

struct GlobalObsConfig {
  bool episode_completion_pct = true;
  bool last_action = true;
  bool last_reward = true;
  bool visitation_counts = false;
};

struct GameConfig {
  size_t num_agents;
  unsigned int max_steps;
  bool episode_truncates;
  ObservationCoord obs_width;
  ObservationCoord obs_height;
  std::vector<std::string> resource_names;
  unsigned int num_observation_tokens;
  GlobalObsConfig global_obs;
  std::vector<std::pair<std::string, std::shared_ptr<ActionConfig>>> actions;  // Ordered list of (name, config) pairs
  std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  float resource_loss_prob = 0.0;
  std::unordered_map<int, std::string> tag_id_map;

  // FEATURE FLAGS
  bool track_movement_metrics = false;
  bool recipe_details_obs = false;
  bool allow_diagonals = false;
  std::unordered_map<std::string, float> reward_estimates = {};

  // Inventory regeneration interval (global check timing)
  unsigned int inventory_regen_interval = 0;  // Interval in timesteps (0 = disabled)

  // Global clipper settings
  std::shared_ptr<ClipperConfig> clipper = nullptr;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_TYPES_HPP_
