#ifndef METTAGRID_METTAGRID_OBJECTS_WALL_HPP_
#define METTAGRID_METTAGRID_OBJECTS_WALL_HPP_

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

  virtual void obs(ObsType* obs) const override {
    const auto offsets = Wall::offsets();
    size_t offset_idx = 0;
    obs[offsets[offset_idx++]] = _type_id;
    obs[offsets[offset_idx++]] = this->hp;
    obs[offsets[offset_idx++]] = this->_swappable;
  }

  static std::vector<uint8_t> offsets() {
    std::vector<uint8_t> ids;
    ids.push_back(ObservationFeature::TypeId);
    ids.push_back(ObservationFeature::Hp);
    ids.push_back(ObservationFeature::Swappable);
    return ids;
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_WALL_HPP_
