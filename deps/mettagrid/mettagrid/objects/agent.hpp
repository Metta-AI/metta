#ifndef AGENT_HPP
#define AGENT_HPP

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
  unsigned char max_items;
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
    this->inventory.resize(InventoryItem::InventoryCount);
    this->max_items = cfg["max_inventory"];
    this->resource_rewards.resize(InventoryItem::InventoryCount);
    for (int i = 0; i < InventoryItem::InventoryCount; i++) {
      this->resource_rewards[i] = rewards[InventoryItemNames[i]];
    }
    this->resource_reward_max.resize(InventoryItem::InventoryCount);
    for (int i = 0; i < InventoryItem::InventoryCount; i++) {
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

  void update_inventory(InventoryItem item, short amount) {
    int current_amount = this->inventory[static_cast<int>(item)];
    int new_amount = current_amount + amount;
    if (new_amount > this->max_items) {
      new_amount = this->max_items;
    }
    if (new_amount < 0) {
      new_amount = 0;
    }

    int delta = new_amount - current_amount;
    this->inventory[static_cast<int>(item)] = new_amount;

    if (delta > 0) {
      this->stats.add(InventoryItemNames[item], "gained", delta);
    } else if (delta < 0) {
      this->stats.add(InventoryItemNames[item], "lost", -delta);
    }

    this->compute_resource_reward(item);
  }

  inline void compute_resource_reward(InventoryItem item) {
    if (this->resource_rewards[static_cast<int>(item)] == 0) {
      return;
    }

    float new_reward = 0;
    for (int i = 0; i < InventoryItem::InventoryCount; i++) {
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

  virtual void obs(ObsType* obs, const std::vector<unsigned int>& offsets) const override {
    obs[offsets[0]] = 1;
    obs[offsets[1]] = group;
    obs[offsets[2]] = hp;
    obs[offsets[3]] = frozen;
    obs[offsets[4]] = orientation;
    obs[offsets[5]] = color;

    for (int i = 0; i < InventoryItem::InventoryCount; i++) {
      obs[offsets[6 + i]] = inventory[i];
    }
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("agent");
    names.push_back("agent:group");
    names.push_back("hp");
    names.push_back("agent:frozen");
    names.push_back("agent:orientation");
    names.push_back("agent:color");

    for (const auto& name : InventoryItemNames) {
      names.push_back("agent:inv:" + name);
    }
    return names;
  }
};

#endif
