#ifndef OBJECTS_WALL_HPP_
#define OBJECTS_WALL_HPP_

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
    this->_swappable = cfg["swappable"];
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.reserve(2);
    features.push_back({ObservationFeature::TypeId, _type_id});
    if (_swappable) {
      // Only emit the token if it's swappable, to reduce the number of tokens.
      features.push_back({ObservationFeature::Swappable, 1});
    }
    return features;
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif  // OBJECTS_WALL_HPP_
