#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

// We want empty tokens to be 0xff, since 0s are very natural numbers to have in the observations, and we want
// empty to be obviously different.
const uint8_t EmptyTokenByte = 0xff;

// ObservationFeature namespace is now defined in config/observation_features.hpp
// and initialized at runtime from the config. This allows dynamic feature configuration.

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_
