#ifndef OBJECTS_WALL_HPP_
#define OBJECTS_WALL_HPP_

#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

// #MettagridConfig
struct WallConfig : public GridObjectConfig {
  WallConfig(TypeId type_id, const std::string& type_name, bool swappable)
      : GridObjectConfig(type_id, type_name), swappable(swappable) {}

  bool swappable;
};

class Wall : public MettaObject {
public:
  bool _swappable;

  Wall(GridCoord r, GridCoord c, const WallConfig& cfg) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::Object_Layer));
    this->_swappable = cfg.swappable;
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.reserve(2);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});

    if (_swappable) {
      // Only emit the swappable observation feature when True to reduce the number of tokens.
      features.push_back({ObservationFeature::Swappable, static_cast<ObservationType>(1)});
    }

    return features;
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif  // OBJECTS_WALL_HPP_
