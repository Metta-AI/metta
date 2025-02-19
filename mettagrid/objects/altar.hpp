#ifndef ALTAR_HPP
#define ALTAR_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

typedef unsigned char ObsType;

class Altar : public Usable {
public:
    Altar(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::AltarT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) && actor->inventory[InventoryItem::battery] > 2;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::battery, -3, rewards);
        actor->update_inventory(InventoryItem::heart, 1, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::battery], "used", 3);
        actor->stats.incr(InventoryItemNames[InventoryItem::heart], "created");
        actor->stats.add(
            InventoryItemNames[InventoryItem::battery],
            "converted",
            InventoryItemNames[InventoryItem::heart], 3);
    }

    inline void obs(ObsType *obs) {
        obs[0] = 1;
        obs[1] = hp;
        obs[2] = ready;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.push_back("altar");
        names.push_back("altar:hp");
        names.push_back("altar:ready");
        return names;
    }
};

#endif
