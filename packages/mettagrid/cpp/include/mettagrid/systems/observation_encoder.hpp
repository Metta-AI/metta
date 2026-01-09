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
                              const std::unordered_map<std::string, ObservationType>& feature_ids,
                              unsigned int token_value_base = 256)
      : protocol_details_obs(protocol_details_obs),
        resource_count(resource_names.size()),
        _token_value_base(token_value_base) {
    // Compute number of tokens needed to encode max uint16_t value (65535)
    _num_inventory_tokens = compute_num_tokens(65535, _token_value_base);

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

    // Build inventory feature ID maps using multi-token encoding
    // inv:{resource} = base token (always emitted)
    // inv:{resource}:p1 = first power token (emitted if amount >= token_value_base)
    // inv:{resource}:p2 = second power token (emitted if amount >= token_value_base^2)
    // etc.
    _inventory_feature_ids.resize(resource_names.size());
    _inventory_power_feature_ids.resize(resource_names.size());

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

      // Power feature IDs (inv:{resource}:p1, inv:{resource}:p2, etc.)
      std::vector<ObservationType> power_ids;
      for (size_t p = 1; p < _num_inventory_tokens; ++p) {
        const std::string power_feature_key = "inv:" + resource_name + ":p" + std::to_string(p);
        auto power_it = feature_ids.find(power_feature_key);
        if (power_it != feature_ids.end()) {
          power_ids.push_back(power_it->second);
        } else {
          throw std::runtime_error("Inventory power feature '" + power_feature_key + "' not found in feature_ids");
        }
      }
      _inventory_power_feature_ids[i] = power_ids;
    }
  }

  static size_t compute_num_tokens(uint32_t max_value, unsigned int base) {
    if (base <= 1) {
      throw std::runtime_error("Base must be greater than 1");
    }
    if (max_value == 0) {
      return 1;
    }
    // Count digits needed: ceil(log_base(max_value + 1))
    size_t count = 0;
    uint32_t value = max_value;
    while (value > 0) {
      value /= base;
      ++count;
    }
    return count;
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

  ObservationType get_inventory_power_feature_id(InventoryItem item, size_t power) const {
    if (item >= _inventory_power_feature_ids.size()) {
      throw std::runtime_error("Invalid item index for inventory power feature lookup");
    }
    if (power == 0 || power > _inventory_power_feature_ids[item].size()) {
      throw std::runtime_error("Invalid power index for inventory power feature lookup");
    }
    return _inventory_power_feature_ids[item][power - 1];
  }

  // Encode inventory amount using multi-token encoding with configurable base.
  // inv:{resource} = amount % token_value_base (always emitted)
  // inv:{resource}:p1 = (amount / token_value_base) % token_value_base (only emitted if amount >= token_value_base)
  // inv:{resource}:p2 = (amount / token_value_base^2) % token_value_base (only emitted if amount >= token_value_base^2)
  // etc.
  void append_inventory_tokens(std::vector<PartialObservationToken>& features,
                               InventoryItem item,
                               InventoryQuantity amount) const {
    // Base token (always emitted)
    ObservationType base_value = static_cast<ObservationType>(amount % _token_value_base);
    features.push_back({_inventory_feature_ids[item], base_value});

    // Higher power tokens (only emitted if needed)
    InventoryQuantity remaining = amount / _token_value_base;
    const auto& power_ids = _inventory_power_feature_ids[item];
    for (size_t p = 0; p < power_ids.size() && remaining > 0; ++p) {
      ObservationType power_value = static_cast<ObservationType>(remaining % _token_value_base);
      features.push_back({power_ids[p], power_value});
      remaining /= _token_value_base;
    }
  }

  unsigned int get_token_value_base() const {
    return _token_value_base;
  }

  size_t get_num_inventory_tokens() const {
    return _num_inventory_tokens;
  }

  bool protocol_details_obs;

private:
  size_t resource_count;
  unsigned int _token_value_base;
  size_t _num_inventory_tokens;
  std::vector<ObservationType> _input_feature_ids;
  std::vector<ObservationType> _output_feature_ids;
  std::vector<ObservationType>
      _inventory_feature_ids;  // Maps item index to base feature ID (amount % token_value_base)
  std::vector<std::vector<ObservationType>> _inventory_power_feature_ids;  // Maps item index to power feature IDs
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
