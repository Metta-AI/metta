#ifndef METTAGRID_METTAGRID_OBJECTS_FREEZE_TILE_HPP_
#define METTAGRID_METTAGRID_OBJECTS_FREEZE_TILE_HPP_

#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class FreezeTile : public MettaObject {
public:
  FreezeTile(GridCoord r, GridCoord c, ObjectConfig cfg) {
    GridObject::init(ObjectType::FreezeTileT, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::init_mo(cfg);
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Hp, hp});
    return features;
  }

  virtual void obs(ObsType* obs, const std::vector<uint8_t>& offsets) const override {
    obs[offsets[0]] = _type_id;
    obs[offsets[1]] = this->hp;
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("type_id");
    names.push_back("hp");
    return names;
  }

  virtual bool swappable() const override {
    return false;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_FREEZE_TILE_HPP_
