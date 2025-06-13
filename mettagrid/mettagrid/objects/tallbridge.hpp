#ifndef METTAGRID_METTAGRID_OBJECTS_TALLBRIDGE_HPP_
#define METTAGRID_METTAGRID_OBJECTS_TALLBRIDGE_HPP_

#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class TallBridge : public MettaObject {
public:
  TallBridge(GridCoord r, GridCoord c, ObjectConfig cfg) {
    GridObject::init(ObjectType::TallBridgeT, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::init_mo(cfg);
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Hp, hp});
    features.push_back({ObservationFeature::ZLevel, 1}); // TallBridge exists at upper Z-level
    return features;
  }

  virtual void obs(ObsType* obs, const std::vector<uint8_t>& offsets) const override {
    obs[offsets[0]] = _type_id;
    obs[offsets[1]] = this->hp;
    obs[offsets[2]] = 1; // Upper Z-level
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("type_id");
    names.push_back("hp");
    names.push_back("z_level");
    return names;
  }

  virtual bool swappable() const override {
    return false;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_TALLBRIDGE_HPP_
