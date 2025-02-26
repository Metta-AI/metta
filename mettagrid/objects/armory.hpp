#ifndef ARMORY_HPP
#define ARMORY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Armory : public Usable {
public:
    Armory(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::ArmoryT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) &&
               actor->inventory[InventoryItem::ore] > 2;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::ore, -3, rewards);
        actor->update_inventory(InventoryItem::armor, 1, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::ore], "used", 3);
        actor->stats.incr(InventoryItemNames[InventoryItem::armor], "created");

        actor->stats.add(
            InventoryItemNames[InventoryItem::ore],
            "converted",
            InventoryItemNames[InventoryItem::armor], 3);
    }

    virtual void obs(ObsType* obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = hp;
        obs[offsets[2]] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("armory");
        features.push_back("hp");
        features.push_back("ready");
        return features;
    }
};

#endif
