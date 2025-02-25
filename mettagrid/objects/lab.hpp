#ifndef LAB_HPP
#define LAB_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Lab : public Usable {
public:
    Lab(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::LabT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) &&
               actor->inventory[InventoryItem::battery] > 2 &&
               actor->inventory[InventoryItem::ore] > 2;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::battery, -3, rewards);
        actor->update_inventory(InventoryItem::ore, -3, rewards);
        actor->update_inventory(InventoryItem::blueprint, 1, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::battery], "used", 3);
        actor->stats.add(InventoryItemNames[InventoryItem::ore], "used", 3);
        actor->stats.incr(InventoryItemNames[InventoryItem::blueprint], "created");

        actor->stats.add(
            InventoryItemNames[InventoryItem::battery],
            "converted",
            InventoryItemNames[InventoryItem::blueprint], 3);
        actor->stats.add(
            InventoryItemNames[InventoryItem::ore],
            "converted",
            InventoryItemNames[InventoryItem::blueprint], 3);
    }

    virtual void obs(ObsType* obs) const override {
        obs[0] = 1;
        obs[1] = hp;
        obs[2] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("lab");
        features.push_back("lab:hp");
        features.push_back("lab:ready");
        return features;
    }
};

#endif
