#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/mettagrid_config.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"

class ObservationEncoder {
public:
  explicit ObservationEncoder(bool protocol_details_obs,
                              const std::vector<std::string>& resource_names,
                              const std::unordered_map<std::string, ObservationType>& feature_ids)
      : protocol_details_obs(protocol_details_obs), resource_count(resource_names.size()) {
    // Build feature ID maps for protocol details if enabled
    if (protocol_details_obs) {
      // Build maps from resource_id -> feature_id for both input and output
      _input_feature_ids.resize(resource_names.size());
      _output_feature_ids.resize(resource_names.size());

      for (size_t i = 0; i < resource_names.size(); ++i) {
        const std::string& resource_name = resource_names[i];

        // Look up input feature ID
        auto input_it = feature_ids.find("protocol_input:" + resource_name);
        if (input_it != feature_ids.end()) {
          _input_feature_ids[i] = input_it->second;
        } else {
          throw std::runtime_error("Protocol input feature 'protocol_input:" + resource_name +
                                   "' not found in feature_ids");
        }

        // Look up output feature ID
        auto output_it = feature_ids.find("protocol_output:" + resource_name);
        if (output_it != feature_ids.end()) {
          _output_feature_ids[i] = output_it->second;
        } else {
          throw std::runtime_error("Protocol output feature 'protocol_output:" + resource_name +
                                   "' not found in feature_ids");
        }
      }
    }

    // Build inventory feature ID maps using exponential encoding (base, e2, e4)
    _inventory_feature_ids.resize(resource_names.size());
    _inventory_e2_feature_ids.resize(resource_names.size());
    _inventory_e4_feature_ids.resize(resource_names.size());
    for (size_t i = 0; i < resource_names.size(); ++i) {
      const std::string& resource_name = resource_names[i];

      // Base feature ID (inv:{resource})
      const std::string feature_key = "inv:" + resource_name;
      auto it = feature_ids.find(feature_key);
      if (it != feature_ids.end()) {
        _inventory_feature_ids[i] = it->second;
      } else {
        throw std::runtime_error("Inventory feature 'inv:" + resource_name + "' not found in feature_ids");
      }

      // e2 feature ID (inv:{resource}:e2)
      const std::string e2_feature_key = "inv:" + resource_name + ":e2";
      auto e2_it = feature_ids.find(e2_feature_key);
      if (e2_it != feature_ids.end()) {
        _inventory_e2_feature_ids[i] = e2_it->second;
      } else {
        throw std::runtime_error("Inventory e2 feature 'inv:" + resource_name + ":e2' not found in feature_ids");
      }

      // e4 feature ID (inv:{resource}:e4)
      const std::string e4_feature_key = "inv:" + resource_name + ":e4";
      auto e4_it = feature_ids.find(e4_feature_key);
      if (e4_it != feature_ids.end()) {
        _inventory_e4_feature_ids[i] = e4_it->second;
      } else {
        throw std::runtime_error("Inventory e4 feature 'inv:" + resource_name + ":e4' not found in feature_ids");
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
  // observer_agent_id: The agent observing this object (UINT_MAX means no specific observer)
  size_t encode_tokens(const GridObject* obj,
                       ObservationTokens tokens,
                       ObservationType location,
                       unsigned int observer_agent_id = UINT_MAX) {
    return append_tokens_if_room_available(tokens, obj->obs_features(observer_agent_id), location);
  }

  size_t get_resource_count() const {
    return resource_count;
  }

  ObservationType get_input_feature_id(size_t resource_id) const {
    if (resource_id >= _input_feature_ids.size()) {
      throw std::runtime_error("Invalid resource_id for input feature lookup");
    }
    return _input_feature_ids[resource_id];
  }

  ObservationType get_output_feature_id(size_t resource_id) const {
    if (resource_id >= _output_feature_ids.size()) {
      throw std::runtime_error("Invalid resource_id for output feature lookup");
    }
    return _output_feature_ids[resource_id];
  }

  ObservationType get_inventory_feature_id(InventoryItem item) const {
    if (item >= _inventory_feature_ids.size()) {
      throw std::runtime_error("Invalid item index for inventory feature lookup");
    }
    return _inventory_feature_ids[item];
  }

  ObservationType get_inventory_e2_feature_id(InventoryItem item) const {
    if (item >= _inventory_e2_feature_ids.size()) {
      throw std::runtime_error("Invalid item index for inventory e2 feature lookup");
    }
    return _inventory_e2_feature_ids[item];
  }

  ObservationType get_inventory_e4_feature_id(InventoryItem item) const {
    if (item >= _inventory_e4_feature_ids.size()) {
      throw std::runtime_error("Invalid item index for inventory e4 feature lookup");
    }
    return _inventory_e4_feature_ids[item];
  }

  // Encode inventory amount using exponential encoding with three tokens.
  // inv:{resource} = amount % 100 (0-99, always emitted)
  // inv:{resource}:e2 = (amount / 100) % 100 (0-99, only emitted if amount >= 100)
  // inv:{resource}:e4 = amount / 10000 (0-100, only emitted if amount >= 10000)
  // This allows representing values up to 1,009,999 (100 * 10000 + 99 * 100 + 99).
  void append_inventory_tokens(std::vector<PartialObservationToken>& features,
                               InventoryItem item,
                               InventoryQuantity amount) const {
    ObservationType base = static_cast<ObservationType>(amount % 100);
    features.push_back({_inventory_feature_ids[item], base});
    if (amount >= 100) {
      InventoryQuantity e2_value = static_cast<InventoryQuantity>((amount / 100) % 100);
      features.push_back({_inventory_e2_feature_ids[item], static_cast<ObservationType>(e2_value)});
    }
    if (amount >= 10000) {
      InventoryQuantity e4_value = static_cast<InventoryQuantity>(amount / 10000);
      ObservationType e4_capped = static_cast<ObservationType>(std::min(e4_value, static_cast<InventoryQuantity>(100)));
      features.push_back({_inventory_e4_feature_ids[item], e4_capped});
    }
  }

  bool protocol_details_obs;

private:
  size_t resource_count;
  std::vector<ObservationType> _input_feature_ids;
  std::vector<ObservationType> _output_feature_ids;
  std::vector<ObservationType> _inventory_feature_ids;     // Maps item index to base feature ID (amount % 100)
  std::vector<ObservationType> _inventory_e2_feature_ids;  // Maps item index to e2 feature ID ((amount / 100) % 100)
  std::vector<ObservationType> _inventory_e4_feature_ids;  // Maps item index to e4 feature ID (amount / 10000)
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
