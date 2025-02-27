#ifndef WALL_HPP
#define WALL_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"
class Wall : public MettaObject {
public:
    Wall(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::WallT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
    }

    virtual void obs(ObsType *obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = this->hp;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.push_back("wall");
        names.push_back("hp");
        return names;
    }
};

#endif
