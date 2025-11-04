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
  explicit ObservationEncoder(size_t resource_count,
                              bool protocol_details_obs = false,
                              const std::vector<std::string>* resource_names = nullptr,
                              const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr)
      : protocol_details_obs(protocol_details_obs), resource_count(resource_count) {
    // Build feature ID maps for protocol details if enabled
    if (protocol_details_obs) {
      if (resource_names && feature_ids && !resource_names->empty()) {
        // Build maps from resource_id -> feature_id for both input and output
        _input_feature_ids.resize(resource_names->size());
        _output_feature_ids.resize(resource_names->size());

        for (size_t i = 0; i < resource_names->size(); ++i) {
          const std::string& resource_name = (*resource_names)[i];

          // Look up input feature ID
          auto input_it = feature_ids->find("input:" + resource_name);
          if (input_it != feature_ids->end()) {
            _input_feature_ids[i] = input_it->second;
          } else {
            throw std::runtime_error("Protocol input feature 'input:" + resource_name + "' not found in feature_ids");
          }

          // Look up output feature ID
          auto output_it = feature_ids->find("output:" + resource_name);
          if (output_it != feature_ids->end()) {
            _output_feature_ids[i] = output_it->second;
          } else {
            throw std::runtime_error("Protocol output feature 'output:" + resource_name + "' not found in feature_ids");
          }
        }
      } else {
        throw std::runtime_error(
            "ObservationEncoder with protocol_details_obs requires resource_names and feature_ids");
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

  bool protocol_details_obs;

private:
  size_t resource_count;
  std::vector<ObservationType> _input_feature_ids;
  std::vector<ObservationType> _output_feature_ids;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_OBSERVATION_ENCODER_HPP_
