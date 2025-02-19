#ifndef MINE_HPP
#define MINE_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

typedef unsigned char ObsType;

class Mine : public Usable {
public:
    Mine(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::MineT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
    }

    inline bool usable(const Agent *actor) override {
        return Usable::usable(actor);
    }

    inline void use(Agent *actor, float *rewards) override {
        actor->update_inventory(InventoryItem::ore, 1, rewards);
        actor->stats.incr(InventoryItemNames[InventoryItem::ore], "created");
    }

    inline void obs(ObsType* obs) const {
        obs[0] = 1;
        obs[1] = hp;
        obs[2] = ready;
    }

    static inline std::vector<std::string> feature_names() {
        std::vector<std::string> features;
        features.push_back("mine");
        features.push_back("mine:hp");
        features.push_back("mine:ready");
        return features;
    }
};

#endif
