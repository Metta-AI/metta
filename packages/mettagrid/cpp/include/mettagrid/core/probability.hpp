#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_PROBABILITY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_PROBABILITY_HPP_

#include <cmath>
#include <random>
#include <stdexcept>

#include "core/types.hpp"

inline InventoryDelta probabilistic_delta(InventoryProbability amount, std::mt19937& rng) {
  if (!std::isfinite(amount)) {
    throw std::runtime_error("probabilistic_delta amount must be finite");
  }
  InventoryProbability magnitude = std::fabs(amount);
  // Guard against overflow when converting to InventoryQuantity/Delta (uint8_t range for quantities)
  if (std::ceil(magnitude) > 255.0f) {
    throw std::runtime_error("probabilistic_delta ceil(|amount|) must be <= 255");
  }

  InventoryQuantity integer_part = static_cast<InventoryQuantity>(std::floor(magnitude));
  InventoryProbability fractional_part =
      magnitude - static_cast<InventoryProbability>(integer_part);

  InventoryDelta delta = static_cast<InventoryDelta>(integer_part);
  if (fractional_part > 0.0f) {
    float sample = std::generate_canonical<float, 10>(rng);
    if (sample < fractional_part) {
      delta = static_cast<InventoryDelta>(delta + 1);
    }
  }

  if (amount < 0.0f) {
    delta = static_cast<InventoryDelta>(-delta);
  }
  return delta;
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_PROBABILITY_HPP_
