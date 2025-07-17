#ifndef OBJECTS_AGENT_HPP_
#define OBJECTS_AGENT_HPP_

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "types.hpp"

// #MettagridConfig
struct AgentConfig : public GridObjectConfig {
  AgentConfig(TypeId type_id,
              const std::string& type_name,
              unsigned char group_id,
              const std::string& group_name,
              unsigned char freeze_duration,
              float action_failure_penalty,
              const std::map<InventoryItem, InventoryQuantity>& resource_limits,
              const std::map<InventoryItem, RewardType>& resource_rewards,
              const std::map<InventoryItem, InventoryQuantity>& resource_reward_max,
              const std::map<std::string, RewardType>& stat_rewards,
              const std::map<std::string, RewardType>& stat_reward_max,
              float group_reward_pct)
      : GridObjectConfig(type_id, type_name),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        resource_limits(resource_limits),
        resource_rewards(resource_rewards),
        resource_reward_max(resource_reward_max),
        stat_rewards(stat_rewards),
        stat_reward_max(stat_reward_max),
        group_reward_pct(group_reward_pct) {}
  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  float action_failure_penalty;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, InventoryQuantity> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  float group_reward_pct;
};

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
  std::map<InventoryItem, InventoryQuantity> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  float action_failure_penalty;
  std::string group_name;
  ObservationType color;
  ObservationType glyph;
  unsigned char agent_id;  // index into MettaGrid._agents (std::vector<Agent*>)
  StatsTracker stats;
  RewardType current_resource_reward;
  RewardType current_stat_reward;
  RewardType* reward;

  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::Up),
        inventory(),  // default constructor
        resource_rewards(config.resource_rewards),
        resource_reward_max(config.resource_reward_max),
        stat_rewards(config.stat_rewards),
        stat_reward_max(config.stat_reward_max),
        action_failure_penalty(config.action_failure_penalty),
        group_name(config.group_name),
        color(0),
        glyph(0),
        agent_id(0),
        stats(),  // default constructor
        current_resource_reward(0),
        current_stat_reward(0),
        reward(nullptr) {
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer));
  }

  void init(RewardType* reward_ptr) {
    this->reward = reward_ptr;
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
      this->stats.add(this->stats.inventory_item_name(item) + ".gained", static_cast<float>(delta));
    } else if (delta < 0) {
      this->stats.add(this->stats.inventory_item_name(item) + ".lost", static_cast<float>(-delta));
    }

    // Update resource rewards incrementally
    this->_update_resource_reward(item, initial_amount, new_amount);

    return delta;
  }

  // Recalculates resource rewards for the agent.
  // item -- the item that was added or removed from the inventory, triggering the reward recalculation.
  inline void compute_resource_reward(InventoryItem item) {
    if (this->resource_rewards.count(item) == 0) {
      // If the item is not in the resource_rewards map, we don't need to recalculate the reward.
      return;
    }

    // Recalculate resource rewards. Note that we're recalculating our total reward, not just the reward for the
    // item that was added or removed.
    // TODO: consider doing this only once per step, and not every time the inventory changes.
    float new_reward = 0;
    for (const auto& [item, amount] : this->inventory) {
      uint8_t max_val = amount;
      if (this->resource_reward_max.count(item) > 0 && max_val > this->resource_reward_max[item]) {
        max_val = this->resource_reward_max[item];
      }
      new_reward += this->resource_rewards[item] * max_val;
    }
    *this->reward += (new_reward - this->current_resource_reward);
    this->current_resource_reward = new_reward;
  }

  // Compute stat-based rewards
  void compute_stat_rewards() {
    if (this->stat_rewards.empty()) {
      return;
    }

    float new_stat_reward = 0;
    auto stat_dict = this->stats.to_dict();

    for (const auto& [stat_name, reward_per_unit] : this->stat_rewards) {
      if (stat_dict.count(stat_name) > 0) {
        float stat_value = stat_dict[stat_name];

        // Apply max limit if configured
        if (this->stat_reward_max.count(stat_name) > 0) {
          float max_reward = this->stat_reward_max[stat_name];
          float total_reward = reward_per_unit * stat_value;
          if (total_reward > max_reward) {
            stat_value = max_reward / reward_per_unit;
          }
        }

        new_stat_reward += reward_per_unit * stat_value;
      }
    }

    // Update the agent's reward with the difference
    float reward_delta = new_stat_reward - this->current_stat_reward;
    if (reward_delta != 0) {
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
  std::map<InventoryItem, InventoryQuantity> resource_limits;

  inline void _update_resource_reward(InventoryItem item, InventoryQuantity old_amount, InventoryQuantity new_amount) {
    // Early exit if this item doesn't contribute to rewards
    auto reward_it = this->resource_rewards.find(item);
    if (reward_it == this->resource_rewards.end()) {
      return;
    }

    // Apply reward caps if they exist
    InventoryQuantity old_capped = old_amount;
    InventoryQuantity new_capped = new_amount;

    auto max_it = this->resource_reward_max.find(item);
    if (max_it != this->resource_reward_max.end()) {
      old_capped = std::min(old_amount, max_it->second);
      new_capped = std::min(new_amount, max_it->second);
    }

    // Calculate only the delta in reward
    float reward_delta = reward_it->second * (new_capped - old_capped);

    // Update both the current reward and the agent's total reward
    this->current_resource_reward += reward_delta;
    *this->reward += reward_delta;
  }
};

#endif  // OBJECTS_AGENT_HPP_
