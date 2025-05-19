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

  size_t obs_tokens(ObservationTokens tokens, const std::vector<unsigned char>& feature_ids) const override {
    vector<uint8_t> basic_token_values = {1, this->hp, this->_swappable};
    size_t max_basic_tokens = tokens.size() > basic_token_values.size() ? basic_token_values.size() : tokens.size();
    size_t tokens_written = 0;
    for (size_t i = 0; i < max_basic_tokens; i++) {
      tokens[i].feature_id = feature_ids[i];
      tokens[i].value = basic_token_values[i];
      tokens_written++;
    }
    return tokens_written;
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
