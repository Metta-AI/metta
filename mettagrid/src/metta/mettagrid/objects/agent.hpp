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
  unsigned char group;
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
  unsigned char agent_id;  // index into MettaGrid._agents (vector<Agent*>)
  StatsTracker stats;
  float current_resource_reward;
  float* reward;

  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::Up),
        resource_limits(config.resource_limits),  // inventory
        resource_rewards(config.resource_rewards),
        resource_reward_max(config.resource_reward_max),
        action_failure_penalty(config.action_failure_penalty),
        group_name(config.group_name),
                color(0),
        glyph(0),
        agent_id(0),
        // stats - default constructed
        current_resource_reward(0),
        reward(nullptr) {
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer));
  }

  void init(float* reward_ptr) {
    this->reward = reward_ptr;
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta) {
    InventoryQuantity initial_amount = this->inventory[item];

    InventoryQuantity new_amount = static_cast<InventoryQuantity>(std::clamp(
        static_cast<int>(initial_amount + attempted_delta), 0, static_cast<int>(this->resource_limits[item])));

    InventoryDelta delta = new_amount - initial_amount;

    if (new_amount > 0) {
      this->inventory[item] = new_amount;
    } else {
      this->inventory.erase(item);
    }

    if (delta > 0) {
      this->stats.add(this->stats.inventory_item_name(item) + ".gained", delta);
    } else if (delta < 0) {
      this->stats.add(this->stats.inventory_item_name(item) + ".lost", -delta);
    }

    this->compute_resource_reward(item);

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

  bool swappable() const override {
    return this->frozen;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    const int num_tokens = this->inventory.size() + 5 + (glyph > 0 ? 1 : 0);

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
      ObservationType item_observation_feature = InventoryFeatureOffset + item;
      features.push_back({item_observation_feature, static_cast<ObservationType>(amount)});
    }

    return features;
  }

private:
  std::map<InventoryItem, InventoryQuantity> resource_limits;
};

#endif  // OBJECTS_AGENT_HPP_
