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
  explicit ObservationEncoder(const std::vector<std::string>& inventory_item_names) {
    _feature_normalizations = FeatureNormalizations;
    _feature_names = FeatureNames;
    assert(_feature_names.size() == InventoryFeatureOffset);
    assert(_feature_names.size() == _feature_normalizations.size());
    for (int i = 0; i < static_cast<int>(inventory_item_names.size()); i++) {
      _feature_normalizations.insert({InventoryFeatureOffset + i, DEFAULT_INVENTORY_NORMALIZATION});
      _feature_names.insert({InventoryFeatureOffset + i, "inv:" + inventory_item_names[i]});
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

  const std::map<ObservationType, float>& feature_normalizations() const {
    return _feature_normalizations;
  }

  const std::map<ObservationType, std::string>& feature_names() const {
    return _feature_names;
  }

private:
  std::map<ObservationType, float> _feature_normalizations;
  std::map<ObservationType, std::string> _feature_names;
};

#endif  // OBSERVATION_ENCODER_HPP_
