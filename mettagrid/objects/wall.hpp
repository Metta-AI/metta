#ifndef WALL_HPP
#define WALL_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "usable.hpp"
#include "agent.hpp"
#include "constants.hpp"

class Wall : public MettaObject {
public:
    Wall(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::WallT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
    }

    void obs(ObsType *obs) const override {
        obs[0] = 1;
        obs[1] = this->hp;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.push_back("wall");
        names.push_back("wall:hp");
        return names;
    }
};

#endif
