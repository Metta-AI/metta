#ifndef LASERY_HPP
#define LASERY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Lasery : public Usable {
public:
    Lasery(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::LaseryT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) &&
               actor->inventory[InventoryItem::ore] > 0 &&
               actor->inventory[InventoryItem::battery] > 1;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::ore, -1, rewards);
        actor->update_inventory(InventoryItem::battery, -2, rewards);
        actor->update_inventory(InventoryItem::laser, 1, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::ore], "used", 1);
        actor->stats.add(InventoryItemNames[InventoryItem::battery], "used", 2);
        actor->stats.incr(InventoryItemNames[InventoryItem::laser], "created");
        actor->stats.add(
            InventoryItemNames[InventoryItem::ore],
            "converted",
            InventoryItemNames[InventoryItem::laser], 1);
        actor->stats.add(
            InventoryItemNames[InventoryItem::battery],
            "converted",
            InventoryItemNames[InventoryItem::laser], 2);
    }

    virtual void obs(ObsType* obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = hp;
        obs[offsets[2]] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("lasery");
        features.push_back("hp");
        features.push_back("ready");
        return features;
    }
};

#endif
