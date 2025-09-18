#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <vector>

#include "systems/stats_tracker.hpp"
#include "objects/agent_config.hpp"
#include "objects/constants.hpp"
#include "core/types.hpp"

class Agent : public GridObject {
public:
  ObservationType group;
  short frozen;
  short freeze_duration;
  Orientation orientation;
  // inventory is a map of item to amount.
  // keys should be deleted when the amount is 0, to keep iteration faster.
  // however, this should not be relied on for correctness.
  std::map<InventoryItem, InventoryQuantity> inventory;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, RewardType> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  float action_failure_penalty;
  std::string group_name;
  ObservationType color;
  ObservationType glyph;
  // Despite being a GridObjectId, this is different from the `id` property.
  // This is the index into MettaGrid._agents (std::vector<Agent*>)
  GridObjectId agent_id;
  StatsTracker stats;
  RewardType current_stat_reward;
  RewardType* reward;
  // Visitation count grid: tracks how many times the agent has visited each position
  std::vector<std::vector<unsigned int>> visitation_grid;
  bool visitation_counts_enabled = false;
  GridLocation prev_location;
  std::string prev_action_name;
  unsigned int steps_without_motion;

  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::North),
        inventory(),
        resource_rewards(config.resource_rewards),
        resource_reward_max(config.resource_reward_max),
        stat_rewards(config.stat_rewards),
        stat_reward_max(config.stat_reward_max),
        resource_limits(config.resource_limits),
        action_failure_penalty(config.action_failure_penalty),
        group_name(config.group_name),
        color(0),
        glyph(0),
        agent_id(0),
        stats(),  // default constructor
        current_stat_reward(0),
        reward(nullptr),
        prev_location(r, c, GridLayer::AgentLayer),
        prev_action_name(""),
        steps_without_motion(0) {
    populate_initial_inventory(config.initial_inventory);
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer));
  }

  void init(RewardType* reward_ptr) {
    this->reward = reward_ptr;
  }

  void populate_initial_inventory(const std::map<InventoryItem, InventoryQuantity>& initial_inventory) {
    for (const auto& [item, amount] : initial_inventory) {
      if (amount > 0) {
        this->inventory[item] = amount;
      }
    }
  }

  void init_visitation_grid(GridCoord height, GridCoord width) {
    visitation_grid.resize(height, std::vector<unsigned int>(width, 0));
    visitation_counts_enabled = true;
  }

  void reset_visitation_counts() {
    for (auto& row : visitation_grid) {
      std::fill(row.begin(), row.end(), 0);
    }
  }

  void increment_visitation_count(GridCoord r, GridCoord c) {
    if (!visitation_counts_enabled) return;

    if (r < static_cast<GridCoord>(visitation_grid.size()) && c < static_cast<GridCoord>(visitation_grid[0].size())) {
      visitation_grid[r][c]++;
    }
  }

  std::array<unsigned int, 5> get_visitation_counts() const {
    std::array<unsigned int, 5> counts = {0, 0, 0, 0, 0};
    if (!visitation_grid.empty()) {
      counts[0] = get_visitation_count(location.r, location.c);  // center

      // Handle potential underflow at map edge
      if (location.r > 0) {
        counts[1] = get_visitation_count(location.r - 1, location.c);  // up
      }
      counts[2] = get_visitation_count(location.r + 1, location.c);  // down

      if (location.c > 0) {
        counts[3] = get_visitation_count(location.r, location.c - 1);  // left
      }
      counts[4] = get_visitation_count(location.r, location.c + 1);  // right
    }
    return counts;
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta) {
    // Get the initial amount (0 if item doesn't exist)
    InventoryQuantity initial_amount = 0;
    auto inv_it = this->inventory.find(item);
    if (inv_it != this->inventory.end()) {
      initial_amount = inv_it->second;
    }

    // Calculate the new amount with clamping
    InventoryQuantity new_amount = static_cast<InventoryQuantity>(std::clamp(
        static_cast<int>(initial_amount + attempted_delta), 0, static_cast<int>(this->resource_limits[item])));

    InventoryDelta delta = new_amount - initial_amount;

    // Update inventory
    if (new_amount > 0) {
      this->inventory[item] = new_amount;
    } else {
      this->inventory.erase(item);
    }

    // Update stats
    if (delta > 0) {
      this->stats.add(this->stats.resource_name(item) + ".gained", delta);
    } else if (delta < 0) {
      this->stats.add(this->stats.resource_name(item) + ".lost", -delta);
    }

    // Update resource rewards incrementally
    this->_update_resource_reward(item, initial_amount, new_amount);

    return delta;
  }

  void compute_stat_rewards() {
    if (this->stat_rewards.empty()) {
      return;
    }

    float new_stat_reward = 0;
    auto stat_dict = this->stats.to_dict();

    for (const auto& [stat_name, reward_per_unit] : this->stat_rewards) {
      if (stat_dict.count(stat_name) > 0) {
        float stat_value = stat_dict[stat_name];

        float stats_reward = stat_value * reward_per_unit;
        if (this->stat_reward_max.count(stat_name) > 0) {
          stats_reward = std::min(stats_reward, this->stat_reward_max.at(stat_name));
        }

        new_stat_reward += stats_reward;
      }
    }

    // Update the agent's reward with the difference
    float reward_delta = new_stat_reward - this->current_stat_reward;
    if (reward_delta != 0.0f) {
      *this->reward += reward_delta;
      this->current_stat_reward = new_stat_reward;
    }
  }

  bool swappable() const override {
    return this->frozen;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    const size_t num_tokens = this->inventory.size() + 5 + (glyph > 0 ? 1 : 0);

    std::vector<PartialObservationToken> features;
    features.reserve(num_tokens);

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(type_id)});
    features.push_back({ObservationFeature::Group, static_cast<ObservationType>(group)});
    features.push_back({ObservationFeature::Frozen, static_cast<ObservationType>(frozen != 0 ? 1 : 0)});
    features.push_back({ObservationFeature::Orientation, static_cast<ObservationType>(orientation)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(color)});
    if (glyph != 0) features.push_back({ObservationFeature::Glyph, static_cast<ObservationType>(glyph)});

    for (const auto& [item, amount] : this->inventory) {
      // inventory should only contain non-zero amounts
      assert(amount > 0);
      auto item_observation_feature = static_cast<ObservationType>(InventoryFeatureOffset + item);
      features.push_back({item_observation_feature, static_cast<ObservationType>(amount)});
    }

    return features;
  }

private:
  inline void _update_resource_reward(InventoryItem item, InventoryQuantity old_amount, InventoryQuantity new_amount) {
    // Early exit if this item doesn't contribute to rewards
    auto reward_it = this->resource_rewards.find(item);
    if (reward_it == this->resource_rewards.end()) {
      return;
    }

    // Calculate the old and new contributions from this item
    float reward_per_item = reward_it->second;
    float old_contribution = reward_per_item * old_amount;
    float new_contribution = reward_per_item * new_amount;

    // Apply per-item cap if it exists
    auto max_it = this->resource_reward_max.find(item);
    if (max_it != this->resource_reward_max.end()) {
      float reward_cap = max_it->second;
      old_contribution = std::min(old_contribution, reward_cap);
      new_contribution = std::min(new_contribution, reward_cap);
    }

    // Update both the current resource reward and the total reward
    float reward_delta = new_contribution - old_contribution;
    *this->reward += reward_delta;
  }

  unsigned int get_visitation_count(GridCoord r, GridCoord c) const {
    if (visitation_grid.empty() || r >= static_cast<GridCoord>(visitation_grid.size()) ||
        c >= static_cast<GridCoord>(visitation_grid[0].size())) {
      return 0;  // Return 0 for out-of-bounds positions
    }
    return visitation_grid[r][c];
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
