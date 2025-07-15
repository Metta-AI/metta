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
              float group_reward_pct)
      : GridObjectConfig(type_id, type_name),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        resource_limits(resource_limits),
        resource_rewards(resource_rewards),
        resource_reward_max(resource_reward_max),
        group_reward_pct(group_reward_pct) {}
  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  float action_failure_penalty;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, InventoryQuantity> resource_reward_max;
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
  float action_failure_penalty;
  std::string group_name;
  ObservationType color;
  ObservationType glyph;
  unsigned char agent_id;  // index into MettaGrid._agents (std::vector<Agent*>)
  StatsTracker stats;
  RewardType current_resource_reward;
  RewardType* reward;

  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::Up),
        inventory(),  // default constructor
        resource_rewards(config.resource_rewards),
        resource_reward_max(config.resource_reward_max),
        action_failure_penalty(config.action_failure_penalty),
        group_name(config.group_name),
        color(0),
        glyph(0),
        agent_id(0),
        stats(),  // default constructor
        current_resource_reward(0),
        reward(nullptr),
        resource_limits(config.resource_limits) {
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
