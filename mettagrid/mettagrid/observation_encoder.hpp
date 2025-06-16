#ifndef METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_
#define METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_

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

  // Returns the number of tokens that were available to write. This will be the number of tokens actually
  // written if there was enough space -- or a greater number if there was not enough space.
  size_t encode_tokens(const GridObject* obj, ObservationTokens tokens, uint8_t location) {
    size_t attempted_tokens_written = obj->obs_tokens(tokens);
    size_t tokens_written = std::min(attempted_tokens_written, tokens.size());

    for (size_t i = 0; i < tokens_written; i++) {
      tokens[i].location = location;
    }

    return attempted_tokens_written;
  }

  void encode(const GridObject* obj, ObsType* obs) {
    obj->obs(obs);
  }

  const std::map<uint8_t, float>& feature_normalizations() const {
    return _feature_normalizations;
  }

private:
  std::map<uint8_t, float> _feature_normalizations;
};

#endif  // METTAGRID_METTAGRID_OBSERVATION_ENCODER_HPP_
