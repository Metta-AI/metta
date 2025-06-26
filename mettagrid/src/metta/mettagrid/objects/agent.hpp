#ifndef OBJECTS_AGENT_HPP_
#define OBJECTS_AGENT_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class Agent : public MettaObject {
public:
  unsigned char group;
  unsigned char frozen;
  unsigned char freeze_duration;
  unsigned char orientation;
  // inventory is a map of item to amount.
  // keys should be deleted when the amount is 0, to keep iteration faster.
  // however, this should not be relied on for correctness.
  std::map<InventoryItem, uint8_t> inventory;
  std::map<InventoryItem, float> resource_rewards;
  std::map<InventoryItem, float> resource_reward_max;
  float action_failure_penalty;
  std::string group_name;
  unsigned char color;
  unsigned char agent_id;
  StatsTracker stats;
  float current_resource_reward;
  float* reward;

  Agent(GridCoord r,
        GridCoord c,
        unsigned char freeze_duration,
        float action_failure_penalty,
        std::map<InventoryItem, uint8_t> max_items_per_type,
        std::map<InventoryItem, float> resource_rewards,
        std::map<InventoryItem, float> resource_reward_max,
        std::string group_name,
        unsigned char group_id,
        const std::vector<std::string>& inventory_item_names)
      : freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        max_items_per_type(max_items_per_type),
        resource_rewards(resource_rewards),
        resource_reward_max(resource_reward_max),
        group(group_id),
        group_name(group_name),
        color(0),
        current_resource_reward(0),
        stats(inventory_item_names) {
    GridObject::init(ObjectType::AgentT, GridLocation(r, c, GridLayer::Agent_Layer));

    this->frozen = 0;
    this->orientation = 0;
    this->reward = nullptr;
  }

  void init(float* reward) {
    this->reward = reward;
  }

  int update_inventory(InventoryItem item, short amount) {
    int current_amount = this->inventory[item];
    int new_amount = current_amount + amount;
    new_amount = std::clamp(new_amount, 0, static_cast<int>(this->max_items_per_type[item]));

    int delta = new_amount - current_amount;
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
      float max_val = static_cast<float>(amount);
      if (this->resource_reward_max.count(item) > 0 && max_val > this->resource_reward_max[item]) {
        max_val = this->resource_reward_max[item];
      }
      new_reward += this->resource_rewards[item] * max_val;
    }
    *this->reward += (new_reward - this->current_resource_reward);
    this->current_resource_reward = new_reward;
  }

  virtual bool swappable() const override {
    return this->frozen;
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.reserve(5 + this->inventory.size());
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Group, group});
    features.push_back({ObservationFeature::Frozen, frozen});
    features.push_back({ObservationFeature::Orientation, orientation});
    features.push_back({ObservationFeature::Color, color});
    for (const auto& [item, amount] : this->inventory) {
      // inventory should only contain non-zero amounts
      assert(amount > 0);
      features.push_back({static_cast<uint8_t>(InventoryFeatureOffset + item), amount});
    }
    return features;
  }

private:
  std::map<InventoryItem, uint8_t> max_items_per_type;
};

#endif  // OBJECTS_AGENT_HPP_
