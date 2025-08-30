
#ifndef OBSERVATION_TOKENS_HPP_
#define OBSERVATION_TOKENS_HPP_

#include "types.hpp"

static constexpr uint8_t OBSERVATION_EMPTY_TOKEN = 0xff;

// NOTE: We use a namespace here to avoid naming collisions:
// - 'TypeId' conflicts with grid_object.hpp::TypeId
// - 'Orientation' conflicts with the enum class Orientation defined above
// The namespace allows us to use these descriptive names without conflicts.
namespace ObservationFeature {
constexpr ObservationType TypeId = 0;
constexpr ObservationType Group = 1;
constexpr ObservationType Hp = 2;
constexpr ObservationType Frozen = 3;
constexpr ObservationType Orientation = 4;
constexpr ObservationType Color = 5;
constexpr ObservationType ConvertingOrCoolingDown = 6;
constexpr ObservationType Swappable = 7;
constexpr ObservationType Glyph = 8;
constexpr ObservationType CoreFeatureCount = 9;  // Number of core features

constexpr ObservationType InventoryOffset = CoreFeatureCount;  // Start of named inventory item names

}  // namespace ObservationFeature

#endif  // OBSERVATION_TOKENS_HPP_
