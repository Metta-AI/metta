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
    std::map<std::string, size_t> features;
    for (size_t type_id = 0; type_id < ObjectType::Count; ++type_id) {
      for (size_t i = 0; i < _type_feature_names[type_id].size(); ++i) {
        std::string feature_name = _type_feature_names[type_id][i];
        if (features.count(feature_name) == 0) {
          features[feature_name] = features.size();
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

    // Print features map
    printf("Features map: \n");
    for (auto& feature : features) {
      printf("%s: %zu\n", feature.first.c_str(), feature.second);
    }
    printf("\n");

    // Print all type feature names
    printf("Type feature names: \n");
    for (size_t i = 0; i < ObjectType::Count; ++i) {
      printf("Type %zu: [", i);
      for (size_t j = 0; j < _type_feature_names[i].size(); ++j) {
        printf("%s, ", _type_feature_names[i][j].c_str());
      }
      printf("]\n");
    }
    printf("\n");

    // Print all offsets
    printf("Offsets: \n");
    for (size_t i = 0; i < _offsets.size(); ++i) {
      printf("Type %zu: ,[", i);
      for (size_t j = 0; j < _offsets[i].size(); ++j) {
        printf("%d, ", _offsets[i][j]);
      }
      printf("]\n");
    }
    printf("\n");
  }

  void encode(const GridObject* obj, ObsType* obs) {
    encode(obj, obs, _offsets[obj->_type_id]);
    // print offsets
    printf("Offsets: ");
    for (size_t i = 0; i < _offsets[obj->_type_id].size(); ++i) {
      printf("%d ", _offsets[obj->_type_id][i]);
    }
    printf("\n");

    // print the first non-zero value of obs
    bool found = false;
    for (size_t i = 0; i < 10; ++i) {
      if (obs[i] != 0) {
        printf("First non-zero value cpp: index %zu, value %d\n", i, (int)obs[i]);
        found = true;
        break;
      }
    }
    if (!found) {
      printf("No non-zero values found cpp\n");
    }
  }

  void encode(const GridObject* obj, ObsType* obs, const std::vector<unsigned int>& offsets) {
    obj->obs(obs, offsets);
  }

  const std::vector<std::string>& feature_names() const {
    return _feature_names;
  }

  const std::vector<std::vector<std::string>>& type_feature_names() const {
    return _type_feature_names;
  }

private:
  std::vector<std::vector<unsigned int>> _offsets;
  std::vector<std::vector<std::string>> _type_feature_names;
  std::vector<std::string> _feature_names;
};

#endif  // METTAGRID_OBSERVATION_ENCODER_HPP
