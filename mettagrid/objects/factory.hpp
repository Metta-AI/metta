#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Factory : public Usable {
public:
    Factory(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::FactoryT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) &&
               actor->inventory[InventoryItem::blueprint] > 0 &&
               actor->inventory[InventoryItem::ore] > 4 &&
               actor->inventory[InventoryItem::battery] > 4;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::blueprint, -1, rewards);
        actor->update_inventory(InventoryItem::ore, -5, rewards);
        actor->update_inventory(InventoryItem::battery, -5, rewards);
        actor->update_inventory(InventoryItem::armor, 5, rewards);
        actor->update_inventory(InventoryItem::laser, 5, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::blueprint], "used", 1);
        actor->stats.add(InventoryItemNames[InventoryItem::armor], "created", 5);
        actor->stats.add(InventoryItemNames[InventoryItem::laser], "created", 5);
    }

    virtual void obs(ObsType* obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = hp;
        obs[offsets[2]] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("factory");
        features.push_back("hp");
        features.push_back("ready");
        return features;
    }
};

#endif
