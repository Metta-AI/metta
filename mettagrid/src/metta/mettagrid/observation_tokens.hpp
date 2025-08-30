
#ifndef OBSERVATION_TOKENS_HPP_
#define OBSERVATION_TOKENS_HPP_

#include "types.hpp"

static constexpr uint8_t OBSERVATION_EMPTY_TOKEN = 0xff;

struct PartialObservationToken {
  ObservationType feature_id = OBSERVATION_EMPTY_TOKEN;
  ObservationType value = OBSERVATION_EMPTY_TOKEN;
};

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  ObservationType location = OBSERVATION_EMPTY_TOKEN;
  ObservationType feature_id = OBSERVATION_EMPTY_TOKEN;
  ObservationType value = OBSERVATION_EMPTY_TOKEN;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3 * sizeof(uint8_t), "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

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
