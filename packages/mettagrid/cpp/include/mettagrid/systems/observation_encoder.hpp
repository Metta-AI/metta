#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"

class ObservationEncoder {
public:
  explicit ObservationEncoder(const std::vector<std::string>& resource_names, bool recipe_details_obs = false)
      : recipe_details_obs(recipe_details_obs), resource_count(resource_names.size()) {
    _feature_normalizations = FeatureNormalizations;
    _feature_names = FeatureNames;
    assert(_feature_names.size() == InventoryFeatureOffset);
    assert(_feature_names.size() == _feature_normalizations.size());

    // Add inventory features
    for (size_t i = 0; i < resource_names.size(); i++) {
      auto observation_feature = InventoryFeatureOffset + static_cast<ObservationType>(i);
      _feature_normalizations.insert({observation_feature, DEFAULT_INVENTORY_NORMALIZATION});
      _feature_names.insert({observation_feature, "inv:" + resource_names[i]});
    }

    if (this->recipe_details_obs) {
      // Define offsets based on actual inventory item count
      const ObservationType input_recipe_offset = InventoryFeatureOffset + static_cast<ObservationType>(resource_count);
      const ObservationType output_recipe_offset = input_recipe_offset + static_cast<ObservationType>(resource_count);

      // Add input recipe features
      for (size_t i = 0; i < resource_names.size(); i++) {
        auto input_feature = input_recipe_offset + static_cast<ObservationType>(i);
        _feature_normalizations.insert({input_feature, DEFAULT_INVENTORY_NORMALIZATION});
        _feature_names.insert({input_feature, "input:" + resource_names[i]});
      }

      // Add output recipe features
      for (size_t i = 0; i < resource_names.size(); i++) {
        auto output_feature = output_recipe_offset + static_cast<ObservationType>(i);
        _feature_normalizations.insert({output_feature, DEFAULT_INVENTORY_NORMALIZATION});
        _feature_names.insert({output_feature, "output:" + resource_names[i]});
      }
    }
  }

  size_t append_tokens_if_room_available(ObservationTokens tokens,
                                         const std::vector<PartialObservationToken>& tokens_to_append,
                                         ObservationType location) {
    size_t tokens_to_write = std::min(tokens.size(), tokens_to_append.size());
    for (size_t i = 0; i < tokens_to_write; i++) {
      tokens[i].location = location;
      tokens[i].feature_id = tokens_to_append[i].feature_id;
      tokens[i].value = tokens_to_append[i].value;
    }
    return tokens_to_append.size();
  }

  // Returns the number of tokens that were available to write. This will be the number of tokens actually
  // written if there was enough space -- or a greater number if there was not enough space.
  size_t encode_tokens(const GridObject* obj, ObservationTokens tokens, ObservationType location) {
    return append_tokens_if_room_available(tokens, obj->obs_features(), location);
  }

  const std::unordered_map<ObservationType, float>& feature_normalizations() const {
    return _feature_normalizations;
  }

  const std::unordered_map<ObservationType, std::string>& feature_names() const {
    return _feature_names;
  }

  size_t get_resource_count() const {
    return resource_count;
  }

  ObservationType get_input_recipe_offset() const {
    return InventoryFeatureOffset + static_cast<ObservationType>(resource_count);
  }

  ObservationType get_output_recipe_offset() const {
    return InventoryFeatureOffset + static_cast<ObservationType>(2 * resource_count);
  }

  bool recipe_details_obs;

private:
  size_t resource_count;
  std::unordered_map<ObservationType, float> _feature_normalizations;
  std::unordered_map<ObservationType, std::string> _feature_names;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
