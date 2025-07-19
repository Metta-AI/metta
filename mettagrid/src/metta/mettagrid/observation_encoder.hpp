#ifndef OBSERVATION_ENCODER_HPP_
#define OBSERVATION_ENCODER_HPP_

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
  explicit ObservationEncoder(const std::vector<std::string>& inventory_item_names, bool recipe_details_obs = false)
      : recipe_details_obs(recipe_details_obs), inventory_item_count(inventory_item_names.size()) {
    _feature_normalizations = FeatureNormalizations;
    _feature_names = FeatureNames;
    assert(_feature_names.size() == InventoryFeatureOffset);
    assert(_feature_names.size() == _feature_normalizations.size());

    // Add inventory features
    for (size_t i = 0; i < inventory_item_names.size(); i++) {
      auto observation_feature = InventoryFeatureOffset + static_cast<ObservationType>(i);
      _feature_normalizations.insert({observation_feature, DEFAULT_INVENTORY_NORMALIZATION});
      _feature_names.insert({observation_feature, "inv:" + inventory_item_names[i]});
    }

    if (this->recipe_details_obs) {
      // Define offsets based on actual inventory item count
      const ObservationType input_recipe_offset = InventoryFeatureOffset + static_cast<ObservationType>(inventory_item_count);
      const ObservationType output_recipe_offset = input_recipe_offset + static_cast<ObservationType>(inventory_item_count);

      // Add input recipe features
      for (size_t i = 0; i < inventory_item_names.size(); i++) {
        auto input_feature = input_recipe_offset + static_cast<ObservationType>(i);
        _feature_normalizations.insert({input_feature, DEFAULT_INVENTORY_NORMALIZATION});
        _feature_names.insert({input_feature, "input:" + inventory_item_names[i]});
      }

      // Add output recipe features
      for (size_t i = 0; i < inventory_item_names.size(); i++) {
        auto output_feature = output_recipe_offset + static_cast<ObservationType>(i);
        _feature_normalizations.insert({output_feature, DEFAULT_INVENTORY_NORMALIZATION});
        _feature_names.insert({output_feature, "output:" + inventory_item_names[i]});
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

  std::map<ObservationType, float> feature_normalizations() const {
    return _feature_normalizations;
  }

  std::map<ObservationType, std::string> feature_names() const {
    return _feature_names;
  }

  size_t get_inventory_item_count() const {
    return inventory_item_count;
  }

  ObservationType get_input_recipe_offset() const {
    return InventoryFeatureOffset + static_cast<ObservationType>(inventory_item_count);
  }

  ObservationType get_output_recipe_offset() const {
    return InventoryFeatureOffset + static_cast<ObservationType>(2 * inventory_item_count);
  }

  bool recipe_details_obs;

private:
  size_t inventory_item_count;
  std::map<ObservationType, float> _feature_normalizations;
  std::map<ObservationType, std::string> _feature_names;
};

#endif  // OBSERVATION_ENCODER_HPP_
