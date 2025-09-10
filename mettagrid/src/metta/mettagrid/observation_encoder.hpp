#ifndef OBSERVATION_ENCODER_HPP_
#define OBSERVATION_ENCODER_HPP_

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "grid_object.hpp"
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
    
    // Compute the first available ID for tags dynamically
    ObservationType first_free_id = InventoryFeatureOffset + static_cast<ObservationType>(resource_count);
    if (recipe_details_obs) {
      first_free_id += static_cast<ObservationType>(resource_count * 2);  // input and output recipes
    }
    _next_tag_feature_id = first_free_id;
    
    // Calculate maximum number of tags we can support
    _max_tags = static_cast<size_t>(std::numeric_limits<ObservationType>::max()) - static_cast<size_t>(first_free_id);
  }
  
  ObservationType register_tag(const std::string& tag) {
    if (tag.empty()) {
      throw std::invalid_argument("Tag cannot be empty");
    }
    
    for (char c : tag) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
        throw std::invalid_argument("Invalid tag name '" + tag + "': tags must contain only alphanumeric characters and underscores");
      }
    }
    
    // Check if tag already registered
    auto existing_it = _tag_feature_ids.find(tag);
    if (existing_it != _tag_feature_ids.end()) {
      return existing_it->second;
    }
    
    if (_tag_feature_ids.size() >= _max_tags) {
      throw std::runtime_error("Maximum number of unique tags (" + std::to_string(_max_tags) + ") exceeded");
    }
    
    if (_next_tag_feature_id >= std::numeric_limits<ObservationType>::max()) {
      throw std::runtime_error("Tag feature ID overflow: cannot allocate more tag feature IDs");
    }
    
    ObservationType next_id = _next_tag_feature_id++;
    auto tag_feature_name = "tag:" + tag;
    _feature_normalizations.insert({next_id, 1.0});
    _feature_names.insert({next_id, tag_feature_name});
    _tag_feature_ids[tag] = next_id;
    return next_id;
  }
  
  ObservationType lookup_tag(const std::string& tag) const {
    auto it = _tag_feature_ids.find(tag);
    if (it == _tag_feature_ids.end()) {
      throw std::runtime_error("Tag '" + tag + "' not found. All tags must be pre-registered before use.");
    }
    return it->second;
  }
  std::vector<ObservationType> get_tag_feature_ids(const std::vector<std::string>& tags) {
    std::vector<ObservationType> feature_ids;
    for (const auto& tag : tags) {
      feature_ids.push_back(lookup_tag(tag));
    }
    return feature_ids;
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

  const std::map<ObservationType, float>& feature_normalizations() const {
    return _feature_normalizations;
  }

  const std::map<ObservationType, std::string>& feature_names() const {
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
  std::map<ObservationType, float> _feature_normalizations;
  std::map<ObservationType, std::string> _feature_names;
  std::map<std::string, ObservationType> _tag_feature_ids;
  ObservationType _next_tag_feature_id;
  size_t _max_tags;
};

#endif  // OBSERVATION_ENCODER_HPP_
