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

  void obs_tokens(ObsType* obs, ObsType prefix, const std::vector<unsigned char>& feature_ids, size_t max_tokens) const override {
    vector<ObsType> basic_token_values = {1, this->hp, this->_swappable};
    size_t max_basic_tokens = max_tokens > basic_token_values.size() ? basic_token_values.size() : max_tokens;
    for (size_t i = 0; i < max_basic_tokens; i++) {
      obs[3 * i] = prefix;
      obs[3 * i + 1] = feature_ids[i];
      obs[3 * i + 2] = basic_token_values[i];
    }
  }

  virtual void obs(ObsType* obs, const std::vector<uint8_t>& offsets) const override {
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
