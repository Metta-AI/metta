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
  ObservationEncoder() {
    _feature_normalizations = FeatureNormalizations;
    for (int i = 0; i < ObservationFeature::ObservationFeatureCount; i++) {
      _feature_normalizations.insert({i, DEFAULT_NORMALIZATION});
    }
    for (int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      _feature_normalizations.insert({static_cast<uint8_t>(InventoryFeatureOffset + i), DEFAULT_NORMALIZATION});
    }
  }

  size_t append_tokens_if_room_available(ObservationTokens tokens,
                                         const std::vector<PartialObservationToken>& tokens_to_append,
                                         uint8_t location) {
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
  size_t encode_tokens(const GridObject* obj, ObservationTokens tokens, uint8_t location) {
    return append_tokens_if_room_available(tokens, obj->obs_features(), location);
  }

  const std::map<uint8_t, float>& feature_normalizations() const {
    return _feature_normalizations;
  }

private:
  std::map<uint8_t, float> _feature_normalizations;
};

#endif  // OBSERVATION_ENCODER_HPP_
