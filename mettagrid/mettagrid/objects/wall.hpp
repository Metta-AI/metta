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
    features.push_back({ObservationFeature::TypeId, scaled_obs(_type_id, ObservationScaling::MediumRange)});
    features.push_back({ObservationFeature::Hp, scaled_obs(hp, ObservationScaling::MediumRange)});
    features.push_back({ObservationFeature::Swappable, scaled_obs(_swappable, ObservationScaling::BooleanRange)});
    return features;
  }

  virtual void obs(ObsType* obs, const std::vector<uint8_t>& offsets) const override {
    obs[offsets[0]] = scaled_obs(_type_id, ObservationScaling::MediumRange);
    obs[offsets[1]] = scaled_obs(hp, ObservationScaling::MediumRange);
    obs[offsets[2]] = scaled_obs(_swappable, ObservationScaling::BooleanRange);
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("type_id");
    names.push_back("hp");
    names.push_back("swappable");
    return names;
  }

  virtual bool swappable() const override {
    return this->_swappable;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_WALL_HPP_
