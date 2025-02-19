#ifndef AGENT_HPP
#define AGENT_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

typedef unsigned char ObsType;

class Agent : public MettaObject {
public:
    unsigned char group;
    unsigned char frozen;
    unsigned char freeze_duration;
    unsigned char orientation;
    std::vector<unsigned char> inventory;
    unsigned char max_items;
    std::vector<float> resource_rewards;
    float action_failure_penalty;
    std::string group_name;
    unsigned char color;
    unsigned char agent_id;
    StatsTracker stats;

    Agent(
        GridCoord r, GridCoord c,
        std::string group_name,
        unsigned char group_id,
        ObjectConfig cfg,
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
        this->action_failure_penalty = rewards["action_failure_penalty"];
        this->color = 0;
    }

    void update_inventory(InventoryItem item, short amount, float *reward) {
        this->inventory[static_cast<int>(item)] += amount;
        if (reward != nullptr) {
            *reward += amount * this->resource_rewards[static_cast<int>(item)];
        }

        if (this->inventory[static_cast<int>(item)] > this->max_items) {
            this->inventory[static_cast<int>(item)] = this->max_items;
        }
        if (this->inventory[static_cast<int>(item)] < 0) {
            this->inventory[static_cast<int>(item)] = 0;
        }

        if (amount > 0) {
            this->stats.add(InventoryItemNames[item], "gained", amount);
        } else {
            this->stats.add(InventoryItemNames[item], "lost", -amount);
        }
    }

    inline void obs(ObsType* obs) const {
        obs[0] = 1;
        obs[1] = group;
        obs[2] = hp;
        obs[3] = frozen;
        obs[4] = orientation;
        obs[5] = color;

        for (int i = 0; i < InventoryItem::InventoryCount; i++) {
            obs[6 + i] = inventory[i];
        }
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.push_back("agent");
        names.push_back("agent:group");
        names.push_back("agent:hp");
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
