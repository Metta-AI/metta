#ifndef METTAGRID_OBSERVATION_ENCODER_HPP
#define METTAGRID_OBSERVATION_ENCODER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"

class ObservationEncoder {
public:
  ObservationEncoder() {
    _offsets.resize(ObjectType::Count);
    _type_feature_names.resize(ObjectType::Count);

    _type_feature_names[ObjectType::AgentT] = Agent::feature_names();
    _type_feature_names[ObjectType::WallT] = Wall::feature_names();

    // These are different types of Converters. The only difference in the feature names
    // is the 1-hot that they use for their type. We're working to simplify this, so we can
    // remove these types from code.
    for (auto type_id : {ObjectType::AltarT,
                         ObjectType::ArmoryT,
                         ObjectType::FactoryT,
                         ObjectType::GeneratorT,
                         ObjectType::LabT,
                         ObjectType::LaseryT,
                         ObjectType::MineT,
                         ObjectType::TempleT}) {
      _type_feature_names[type_id] = Converter::feature_names(type_id);
    }

    // Generate an offset for each unique feature name.
    std::map<std::string, uint8_t> features;

    for (size_t type_id = 0; type_id < ObjectType::Count; ++type_id) {
      for (size_t i = 0; i < _type_feature_names[type_id].size(); ++i) {
        std::string feature_name = _type_feature_names[type_id][i];
        if (features.count(feature_name) == 0) {
          size_t index = features.size();
          // We want to keep the index within the range of a byte since we plan to
          // use this as a feature_id.
          assert(index < 256);
          features.insert({feature_name, index});
          _feature_names.push_back(feature_name);
        }
      }
    }

    // Set the offset for each feature, using the global offsets.
    for (size_t type_id = 0; type_id < ObjectType::Count; ++type_id) {
      for (size_t i = 0; i < _type_feature_names[type_id].size(); ++i) {
        _offsets[type_id].push_back(features[_type_feature_names[type_id][i]]);
      }
    }
  }

  void encode(const GridObject* obj, ObsType* obs) {
    encode(obj, obs, _offsets[obj->_type_id]);
  }

  void encode(const GridObject* obj, ObsType* obs, const std::vector<uint8_t>& offsets) {
    obj->obs(obs, offsets);
  }

  const std::vector<std::string>& feature_names() const {
    return _feature_names;
  }

  const std::vector<std::vector<std::string>>& type_feature_names() const {
    return _type_feature_names;
  }

private:
  std::vector<std::vector<uint8_t>> _offsets;
  std::vector<std::vector<std::string>> _type_feature_names;
  std::vector<std::string> _feature_names;
};

#endif  // METTAGRID_OBSERVATION_ENCODER_HPP
