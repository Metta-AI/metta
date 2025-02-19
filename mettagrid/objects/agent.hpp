#ifndef AGENT_HPP
#define AGENT_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class Agent : public MettaObject {
public:
    unsigned char group;
    unsigned char frozen;
    unsigned char attack_damage;
    unsigned char freeze_duration;
    unsigned char energy;
    unsigned char orientation;
    unsigned char shield;
    unsigned char shield_upkeep;
    std::vector<unsigned char> inventory;
    unsigned char max_items;
    unsigned char max_energy;
    float energy_reward;
    float resource_reward;
    float freeze_reward;
    std::string group_name;
    unsigned char color;
    unsigned char agent_id;
    StatsTracker stats;

    Agent(
        GridCoord r, GridCoord c,
        std::string group_name,
        unsigned char group_id,
        ObjectConfig cfg) {
        GridObject::init(ObjectType::AgentT, GridLocation(r, c, GridLayer::Agent_Layer));
        MettaObject::init_mo(cfg);

        this->group_name = group_name;
        this->group = group_id;
        this->frozen = 0;
        this->attack_damage = cfg["attack_damage"];
        this->freeze_duration = cfg["freeze_duration"];
        this->max_energy = cfg["max_energy"];
        this->energy = 0;
        this->update_energy(cfg["initial_energy"], nullptr);
        this->shield_upkeep = cfg["upkeep.shield"];
        this->orientation = 0;
        this->inventory.resize(InventoryItem::InventoryCount);
        this->max_items = cfg["max_inventory"];
        this->energy_reward = float(cfg["energy_reward"]) / 1000.0;
        this->resource_reward = float(cfg["resource_reward"]) / 1000.0;
        this->freeze_reward = float(cfg["freeze_reward"]) / 1000.0;
        this->shield = false;
        this->color = 0;
    }

    void update_inventory(InventoryItem item, short amount, float *reward) {
        this->inventory[static_cast<int>(item)] += amount;
        if (reward != nullptr && amount > 0) {
            *reward += amount * this->resource_reward;
        }

        if (this->inventory[static_cast<int>(item)] > this->max_items) {
            this->inventory[static_cast<int>(item)] = this->max_items;
        }

        if (amount > 0) {
            this->stats.add(InventoryItemNames[item], "gained", amount);
            this->stats.add(InventoryItemNames[item], "gained", this->group_name, amount);
        } else {
            this->stats.add(InventoryItemNames[item], "lost", -amount);
            this->stats.add(InventoryItemNames[item], "lost", this->group_name, -amount);
        }
    }

    short update_energy(short amount, float *reward) {
        if (amount < 0) {
            if (amount < -this->energy) {
                amount = -this->energy;
            }
        } else {
            if (amount > this->max_energy - this->energy) {
                amount = this->max_energy - this->energy;
            }
        }

        this->energy += amount;
        if (reward != nullptr && amount > 0) {
            *reward += amount * this->energy_reward;
        }

        this->stats.add("energy.gained", amount);
        this->stats.add("energy.gained", this->group_name, amount);

        return amount;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names = {
            "agent",
            "agent:group", 
            "agent:hp",
            "agent:frozen",
            "agent:energy",
            "agent:orientation",
            "agent:shield",
            "agent:color"
        };
        
        for (const auto& name : InventoryItemNames) {
            names.push_back("agent:inv:" + name);
        }
        return names;
    }
};

#endif
