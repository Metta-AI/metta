#ifndef WALL_HPP
#define WALL_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "constants.hpp"
#include "grid_object.hpp"
#include "objects/metta_object.hpp"

class Wall : public MettaObject {
public:
  bool _swappable;

  Wall(GridCoord r, GridCoord c, ObjectConfig cfg) {
    GridObject::init(ObjectType::WallT, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::set_hp(cfg);
    this->_swappable = cfg["swappable"];
  }

  virtual void obs(c_observations_type* obs) const override {
    MettaObject::obs(obs);
    // wall-specific features
    encode(obs, GridFeature::WALL, 1);
    encode(obs, GridFeature::SWAPPABLE, this->_swappable);
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif