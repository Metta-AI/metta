#ifndef AGENT_HPP
#define AGENT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "constants.hpp"
#include "grid_object.hpp"
#include "objects/metta_object.hpp"
#include "stats_tracker.hpp"

class Agent : public HasInventory {
public:
  uint8_t group;
  uint8_t frozen;
  uint8_t freeze_duration;
  uint8_t orientation;
  std::vector<uint8_t> inventory;
  uint8_t max_items;
  std::vector<float> resource_rewards;
  std::vector<float> resource_reward_max;
  float action_failure_penalty;
  std::string group_name;
  uint8_t color;
  uint8_t agent_id;
  StatsTracker stats;
  float current_resource_reward;
  float* reward;

  Agent(GridCoord r,
        GridCoord c,
        std::string group_name,
        uint8_t group_id,
        ObjectConfig cfg,
        // Configuration -- rewards that the agent will get for certain
        // actions or inventory changes.
        std::map<std::string, float> rewards) {
    GridObject::init(ObjectType::AgentT, GridLocation(r, c, GridLayer::Agent_Layer));
    MettaObject::set_hp(cfg);

    this->group_name = group_name;
    this->group = group_id;
    this->frozen = 0;
    this->freeze_duration = cfg["freeze_duration"];
    this->orientation = 0;
    this->inventory.resize(InventoryItem::InventoryCount);
    this->max_items = cfg["max_inventory"];
    this->resource_rewards.resize(InventoryItem::InventoryCount);
    for (int32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      this->resource_rewards[i] = rewards[InventoryItemNames[i]];
    }
    this->resource_reward_max.resize(InventoryItem::InventoryCount);
    for (int32_t i = 0; i < InventoryItem::InventoryCount; i++) {
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

  void update_inventory(InventoryItem item, int16_t amount) {
    int32_t current_amount = this->inventory[static_cast<int32_t>(item)];
    int32_t new_amount = current_amount + amount;
    if (new_amount > this->max_items) {
      new_amount = this->max_items;
    }
    if (new_amount < 0) {
      new_amount = 0;
    }

    int32_t delta = new_amount - current_amount;
    this->inventory[static_cast<int32_t>(item)] = new_amount;

    if (delta > 0) {
      this->stats.add(InventoryItemNames[item], "gained", delta);
    } else if (delta < 0) {
      this->stats.add(InventoryItemNames[item], "lost", -delta);
    }

    this->compute_resource_reward(item);
  }

  inline void compute_resource_reward(InventoryItem item) {
    if (this->resource_rewards[static_cast<int32_t>(item)] == 0) {
      return;
    }

    float new_reward = 0;
    for (int32_t i = 0; i < InventoryItem::InventoryCount; i++) {
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

  virtual void obs(ObsType* obs) const override {
    HasInventory::obs(obs);

    // Agent-specific features
    encode(obs, "agent", 1);
    encode(obs, "agent:group", group);
    encode(obs, "agent:frozen", frozen);
    encode(obs, "agent:orientation", orientation);
    encode(obs, "agent:color", color);

    // Inventory features
    for (int32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      encode(obs, "agent:inv:" + InventoryItemNames[i], inventory[i]);
    }
  }
};

#endif