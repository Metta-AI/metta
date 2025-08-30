#ifndef OBSERVATION_ENCODER_HPP_
#define OBSERVATION_ENCODER_HPP_

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/wall.hpp"
#include "observation_tokens.hpp"
#include "types.hpp"

class ObservationEncoder {
public:
  static constexpr uint8_t EmptyTokenByte = 0xff;

  explicit ObservationEncoder(const std::vector<std::string>& inventory_item_names)
      : inventory_item_count_(inventory_item_names.size()) {
    InitializeCoreFeatures(inventory_item_names);
  }

  // Get feature ID by name - throws if not found
  ObservationType get_feature(const std::string& feature_name) const {
    auto it = feature_name_to_id_.find(feature_name);
    if (it == feature_name_to_id_.end()) {
      throw std::runtime_error("Unknown feature: " + feature_name);
    }
    return it->second;
  }

  // Check if a feature exists
  bool has_feature(const std::string& feature_name) const {
    return feature_name_to_id_.find(feature_name) != feature_name_to_id_.end();
  }

  // Get normalization value for a feature
  float get_normalization(const std::string& feature_name) const {
    return feature_normalizations_.at(get_feature(feature_name));
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

  size_t encode_tokens(const GridObject* obj, ObservationTokens tokens, ObservationType location) {
    return append_tokens_if_room_available(tokens, obj->obs_features(), location);
  }

  const std::map<ObservationType, float>& feature_normalizations() const {
    return feature_normalizations_;
  }

  const std::map<ObservationType, std::string>& feature_names() const {
    return feature_names_;
  }

  size_t get_inventory_item_count() const {
    return inventory_item_count_;
  }

  ObservationType get_inventory_offset() const {
    return inventory_offset_;
  }

  ObservationType get_input_recipe_offset() const {
    return inventory_offset_ + static_cast<ObservationType>(inventory_item_count_);
  }

  ObservationType get_output_recipe_offset() const {
    return inventory_offset_ + static_cast<ObservationType>(2 * inventory_item_count_);
  }

  ObservationType register_feature(const std::string& name, float normalization = 1.0) {
    ObservationType id = _next_id++;
    RegisterFeature(id, name, normalization);
    return id;
  }

private:
  ObservationType _next_id;

  void InitializeCoreFeatures(const std::vector<std::string>& inventory_item_names) {
    // Initialize the core features values using predefined constants
    RegisterFeature(ObservationFeature::TypeId, "type_id", 1.0);
    RegisterFeature(ObservationFeature::Group, "agent:group", 10.0);
    RegisterFeature(ObservationFeature::Hp, "hp", 30.0);
    RegisterFeature(ObservationFeature::Frozen, "agent:frozen", 1.0);
    RegisterFeature(ObservationFeature::Orientation, "agent:orientation", 1.0);
    RegisterFeature(ObservationFeature::Color, "agent:color", 255.0);
    RegisterFeature(ObservationFeature::ConvertingOrCoolingDown, "converting", 1.0);
    RegisterFeature(ObservationFeature::Swappable, "swappable", 1.0);
    RegisterFeature(ObservationFeature::Glyph, "agent:glyph", 255.0);

    // Inventory features are assumed to follow the core features
    ObservationType inventory_count = static_cast<ObservationType>(inventory_item_names.size());
    for (ObservationType i = 0; i < inventory_count; i++) {
      RegisterFeature(
          ObservationFeature::InventoryOffset + i, "inv:" + inventory_item_names[i], DEFAULT_INVENTORY_NORMALIZATION);
    }

    _next_id = ObservationFeature::CoreFeatureCount + inventory_count;
  }

  void RegisterFeature(ObservationType id, const std::string& name, float normalization) {
    feature_names_[id] = name;
    feature_name_to_id_[name] = id;
    feature_normalizations_[id] = normalization;
  }

  static constexpr float DEFAULT_INVENTORY_NORMALIZATION = 100.0;

  size_t inventory_item_count_;
  ObservationType inventory_offset_;

  std::map<ObservationType, std::string> feature_names_;
  std::map<std::string, ObservationType> feature_name_to_id_;  // Reverse lookup
  std::map<ObservationType, float> feature_normalizations_;
};

#endif  // OBSERVATION_ENCODER_HPP_
