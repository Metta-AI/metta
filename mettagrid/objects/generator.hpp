#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Generator : public Usable {
public:
    Generator(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::GeneratorT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) && actor->inventory[InventoryItem::ore] > 0;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::ore, -1, rewards);
        actor->update_inventory(InventoryItem::battery, 1, rewards);

        actor->stats.incr(InventoryItemNames[InventoryItem::ore], "used");
        actor->stats.incr(
            InventoryItemNames[InventoryItem::ore],
            "converted",
            InventoryItemNames[InventoryItem::battery]);

        actor->stats.incr(InventoryItemNames[InventoryItem::battery], "created");
    }

    void obs(ObsType *obs) const override {
        obs[0] = 1;
        obs[1] = this->hp;
        obs[2] = this->ready;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.push_back("generator");
        names.push_back("generator:hp");
        names.push_back("generator:ready");
        return names;
    }
};

#endif
