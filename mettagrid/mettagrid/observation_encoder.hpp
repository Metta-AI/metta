#ifndef METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_
#define METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_

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
    _offsets.resize(ObjectType::ObjectTypeCount);
    _type_feature_names.resize(ObjectType::ObjectTypeCount);

    _type_feature_names[ObjectType::AgentT] = Agent::feature_names();
    _type_feature_names[ObjectType::WallT] = Wall::feature_names();

    // These are different types of Converters. They all have the same feature names,
    // so this is somewhat redundant.
    for (auto type_id : {ObjectType::AltarT,
                         ObjectType::ArmoryT,
                         ObjectType::FactoryT,
                         ObjectType::GeneratorT,
                         ObjectType::LabT,
                         ObjectType::LaseryT,
                         ObjectType::MineT,
                         ObjectType::TempleT}) {
      _type_feature_names[type_id] = Converter::feature_names();
    }

    // Generate an offset for each unique feature name.
    std::map<std::string, uint8_t> features;

    for (size_t type_id = 0; type_id < ObjectType::ObjectTypeCount; ++type_id) {
      for (size_t i = 0; i < _type_feature_names[type_id].size(); ++i) {
        std::string feature_name = _type_feature_names[type_id][i];
        if (features.count(feature_name) == 0) {
          size_t index = features.size();
          // We want to keep the index within the range of a byte since we plan to
          // use this as a feature_id.
          assert(index < 256);
          features.insert({feature_name, index});
          if (FeatureNormalizations.count(feature_name) > 0) {
            _feature_normalizations.push_back(FeatureNormalizations.at(feature_name));
          } else {
            _feature_normalizations.push_back(DEFAULT_NORMALIZATION);
          }
        }
      }
    }

    // Set the offset for each feature, using the global offsets.
    for (size_t type_id = 0; type_id < ObjectType::ObjectTypeCount; ++type_id) {
      for (size_t i = 0; i < _type_feature_names[type_id].size(); ++i) {
        _offsets[type_id].push_back(features[_type_feature_names[type_id][i]]);
      }
    }
  }

  // Returns the number of tokens that were available to write. This will be the number of tokens actually
  // written if there was enough space -- or a greater number if there was not enough space.
  size_t encode_tokens(const GridObject* obj, ObservationTokens tokens) {
    return obj->obs_tokens(tokens);
  }

  void encode(const GridObject* obj, ObsType* obs) {
    encode(obj, obs, _offsets[obj->_type_id]);
  }

  void encode(const GridObject* obj, ObsType* obs, const std::vector<uint8_t>& offsets) {
    obj->obs(obs, offsets);
  }

  const std::vector<float>& feature_normalizations() const {
    return _feature_normalizations;
  }

  const std::vector<std::vector<std::string>>& type_feature_names() const {
    return _type_feature_names;
  }

private:
  std::vector<std::vector<uint8_t>> _offsets;
  std::vector<std::vector<std::string>> _type_feature_names;
  std::vector<float> _feature_normalizations;
};

#endif  // METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_
