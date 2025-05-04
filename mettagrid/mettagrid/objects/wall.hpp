#ifndef WALL_HPP
#define WALL_HPP

#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class Wall : public MettaObject {
public:
  bool _swappable;

  Wall(GridCoord r, GridCoord c, ObjectConfig cfg) {
    GridObject::init(ObjectType::WallT, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::init_mo(cfg);
    this->_swappable = cfg["swappable"];
  }

  virtual void obs(ObsType* obs, const std::vector<unsigned int>& offsets) const override {
    obs[offsets[0]] = 1;
    obs[offsets[1]] = this->hp;
    obs[offsets[2]] = this->_swappable;
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("wall");
    names.push_back("hp");
    names.push_back("swappable");
    return names;
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif
