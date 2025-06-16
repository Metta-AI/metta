#ifndef METTAGRID_METTAGRID_OBJECTS_AGENT_HPP_
#define METTAGRID_METTAGRID_OBJECTS_AGENT_HPP_

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
        std::string group_name,
        unsigned char group_id,
        ObjectConfig cfg,
        // Configuration -- rewards that the agent will get for certain
        // actions or inventory changes.
        std::map<std::string, float> rewards) {
    GridObject::init(ObjectType::AgentT, GridLocation(r, c, GridLayer::Agent_Layer));
    MettaObject::init_mo(cfg);

    this->group_name = group_name;
    this->group = group_id;
    this->frozen = 0;
    this->freeze_duration = cfg["freeze_duration"];
    this->orientation = 0;
    this->inventory.resize(InventoryItem::InventoryItemCount);
    unsigned char default_item_max = cfg["default_item_max"];
    this->max_items_per_type.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (cfg.find(InventoryItemNames[i] + "_max") != cfg.end()) {
        this->max_items_per_type[i] = cfg[InventoryItemNames[i] + "_max"];
      } else {
        this->max_items_per_type[i] = default_item_max;
      }
    }
    this->resource_rewards.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->resource_rewards[i] = rewards[InventoryItemNames[i]];
    }
    this->resource_reward_max.resize(InventoryItem::InventoryItemCount);
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->resource_reward_max[i] = rewards[InventoryItemNames[i] + "_max"];
    }
    this->action_failure_penalty = rewards["action_failure_penalty"];
    this->color = 0;
    this->current_resource_reward = 0;
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

  virtual void obs(ObsType* obs) const override {
    const auto offsets = Agent::offsets();
    size_t offset_idx = 0;
    obs[offsets[offset_idx++]] = _type_id;
    obs[offsets[offset_idx++]] = group;
    obs[offsets[offset_idx++]] = hp;
    obs[offsets[offset_idx++]] = frozen;
    obs[offsets[offset_idx++]] = orientation;
    obs[offsets[offset_idx++]] = color;

    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      obs[offsets[offset_idx++]] = inventory[i];
    }
  }

  static std::vector<uint8_t> offsets() {
    std::vector<uint8_t> names;
    names.push_back(ObservationFeature::TypeId);
    names.push_back(ObservationFeature::Group);
    names.push_back(ObservationFeature::Hp);
    names.push_back(ObservationFeature::Frozen);
    names.push_back(ObservationFeature::Orientation);
    names.push_back(ObservationFeature::Color);

    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      names.push_back(static_cast<uint8_t>(InventoryFeatureOffset + i));
    }
    return names;
  }

private:
  std::vector<unsigned char> max_items_per_type;
};

#endif  // METTAGRID_METTAGRID_OBJECTS_AGENT_HPP_
