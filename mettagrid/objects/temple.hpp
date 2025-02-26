#ifndef TEMPLE_HPP
#define TEMPLE_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Temple : public Usable {
public:
    Temple(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::TempleT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor) &&
               actor->inventory[InventoryItem::heart] > 0 &&
               actor->inventory[InventoryItem::blueprint] > 0;
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::heart, -1, rewards);
        actor->update_inventory(InventoryItem::blueprint, -1, rewards);
        actor->update_inventory(InventoryItem::heart, 5, rewards);

        actor->stats.add(InventoryItemNames[InventoryItem::heart], "used", 1);
        actor->stats.add(InventoryItemNames[InventoryItem::blueprint], "used", 1);
        actor->stats.add(InventoryItemNames[InventoryItem::heart], "created", 5);
    }

    virtual void obs(ObsType* obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = hp;
        obs[offsets[2]] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("temple");
        features.push_back("hp");
        features.push_back("ready");
        return features;
    }
};

#endif
