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
  std::vector<unsigned char> inventory;
  std::vector<float> resource_rewards;
  std::vector<float> resource_reward_max;
  float action_failure_penalty;
  std::string group_name;
  unsigned char color;
  unsigned char agent_id;
  StatsTracker stats;
  float current_resource_reward;
  float* reward;

  Agent(GridCoord r,
        GridCoord c,
        unsigned char default_item_max,
        unsigned char freeze_duration,
        float action_failure_penalty,
        std::map<std::string, unsigned int> max_items_per_type_,
        std::map<std::string, float> resource_rewards_,
        std::map<std::string, float> resource_reward_max_,
        std::string group_name,
        unsigned char group_id)
      : freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        group(group_id),
        group_name(group_name),
        color(0),
        current_resource_reward(0) {
    GridObject::init(ObjectType::AgentT, GridLocation(r, c, GridLayer::Agent_Layer));

    this->frozen = 0;
    this->orientation = 0;
    this->inventory.resize(InventoryItem::InventoryItemCount);
    this->max_items_per_type.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (max_items_per_type_.find(InventoryItemNames[i]) != max_items_per_type_.end()) {
        this->max_items_per_type[i] = max_items_per_type_[InventoryItemNames[i]];
      } else {
        this->max_items_per_type[i] = default_item_max;
      }
    }
    this->resource_rewards.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->resource_rewards[i] = resource_rewards_[InventoryItemNames[i]];
    }
    this->resource_reward_max.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->resource_reward_max[i] = resource_reward_max_[InventoryItemNames[i]];
    }
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
    this->inventory[item] = new_amount;

    if (delta > 0) {
      this->stats.add(InventoryItemNames[item] + ".gained", delta);
    } else if (delta < 0) {
      this->stats.add(InventoryItemNames[item] + ".lost", -delta);
    }

    this->compute_resource_reward(item);

    return delta;
  }

  inline void compute_resource_reward(InventoryItem item) {
    if (this->resource_rewards[item] == 0) {
      return;
    }

    float new_reward = 0;
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      float max_val = static_cast<float>(this->inventory[i]);
      if (max_val > this->resource_reward_max[i]) {
        max_val = this->resource_reward_max[i];
      }
      new_reward += this->resource_rewards[i] * max_val;
    }
    *this->reward += (new_reward - this->current_resource_reward);
    this->current_resource_reward = new_reward;
  }

  virtual bool swappable() const override {
    return this->frozen;
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.reserve(5 + InventoryItem::InventoryItemCount);
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Group, group});
    features.push_back({ObservationFeature::Frozen, frozen});
    features.push_back({ObservationFeature::Orientation, orientation});
    features.push_back({ObservationFeature::Color, color});
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (inventory[i] > 0) {
        features.push_back({static_cast<uint8_t>(InventoryFeatureOffset + i), inventory[i]});
      }
    }
    return features;
  }

private:
  std::vector<unsigned char> max_items_per_type;
};

#endif  // OBJECTS_AGENT_HPP_
