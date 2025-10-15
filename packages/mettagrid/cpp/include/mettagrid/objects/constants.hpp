#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

enum EventType {
  FinishConverting = 0,
  CoolDown = 1,
  EventTypeCount
};

namespace GridLayer {
constexpr ObservationType AgentLayer = 0;
constexpr ObservationType ObjectLayer = 1;
constexpr ObservationType GridLayerCount = 2;
};  // namespace GridLayer

// We want empty tokens to be 0xff, since 0s are very natural numbers to have in the observations, and we want
// empty to be obviously different.
const uint8_t EmptyTokenByte = 0xff;

// Changing observation feature ids will break models that have
// been trained on the old feature ids.
// In the future, the string -> id mapping should be stored on a
// per-policy basis.
//
// NOTE: We use a namespace here to avoid naming collisions:
// - 'TypeId' conflicts with grid_object.hpp::TypeId
// - 'Orientation' conflicts with the enum class Orientation defined above
// The namespace allows us to use these descriptive names without conflicts.
namespace ObservationFeature {
constexpr ObservationType TypeId = 0;
constexpr ObservationType Group = 1;
constexpr ObservationType Frozen = 2;
constexpr ObservationType Orientation = 3;
// This used to be Color, but we removed it. We're leaving this here for backward compatibility, but may fill it
// in with a more useful feature in the future.
constexpr ObservationType ReservedForFutureUse = 4;
constexpr ObservationType ConvertingOrCoolingDown = 5;
constexpr ObservationType Swappable = 6;
constexpr ObservationType EpisodeCompletionPct = 7;
constexpr ObservationType LastAction = 8;
constexpr ObservationType LastActionArg = 9;
constexpr ObservationType LastReward = 10;
constexpr ObservationType Glyph = 11;
constexpr ObservationType VisitationCounts = 12;
constexpr ObservationType Tag = 13;
constexpr ObservationType CooldownRemaining = 14;
constexpr ObservationType Clipped = 15;
constexpr ObservationType RemainingUses = 16;
constexpr ObservationType ObservationFeatureCount = 17;
}  // namespace ObservationFeature

const ObservationType InventoryFeatureOffset = ObservationFeature::ObservationFeatureCount;

// Use function-local statics to avoid global constructors and initialization order issues
inline const std::unordered_map<ObservationType, std::string>& GetFeatureNames() {
  static const std::unordered_map<ObservationType, std::string> feature_names = {
      {ObservationFeature::TypeId, "type_id"},
      {ObservationFeature::Group, "agent:group"},
      {ObservationFeature::Frozen, "agent:frozen"},
      {ObservationFeature::Orientation, "agent:orientation"},
      {ObservationFeature::ReservedForFutureUse, "agent:reserved_for_future_use"},
      {ObservationFeature::ConvertingOrCoolingDown, "converting"},
      {ObservationFeature::Swappable, "swappable"},
      {ObservationFeature::EpisodeCompletionPct, "episode_completion_pct"},
      {ObservationFeature::LastAction, "last_action"},
      {ObservationFeature::LastActionArg, "last_action_arg"},
      {ObservationFeature::LastReward, "last_reward"},
      {ObservationFeature::Glyph, "agent:glyph"},
      {ObservationFeature::VisitationCounts, "agent:visitation_counts"},
      {ObservationFeature::Tag, "tag"},
      {ObservationFeature::CooldownRemaining, "cooldown_remaining"},
      {ObservationFeature::Clipped, "clipped"},
      {ObservationFeature::RemainingUses, "remaining_uses"}};
  return feature_names;
}

// ##ObservationNormalization
// These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
// but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
inline const std::unordered_map<ObservationType, float>& GetFeatureNormalizations() {
  static const std::unordered_map<ObservationType, float> feature_normalizations = {
      {ObservationFeature::LastAction, 10.0},
      {ObservationFeature::LastActionArg, 10.0},
      {ObservationFeature::EpisodeCompletionPct, 255.0},
      {ObservationFeature::LastReward, 100.0},
      {ObservationFeature::TypeId, 1.0},
      {ObservationFeature::Group, 10.0},
      {ObservationFeature::Frozen, 1.0},
      {ObservationFeature::Orientation, 1.0},
      {ObservationFeature::ReservedForFutureUse, 255.0},
      {ObservationFeature::ConvertingOrCoolingDown, 1.0},
      {ObservationFeature::Swappable, 1.0},
      {ObservationFeature::Glyph, 255.0},
      {ObservationFeature::VisitationCounts, 1000.0},
      {ObservationFeature::Tag, 10.0},
      {ObservationFeature::CooldownRemaining, 255.0},
      {ObservationFeature::Clipped, 1.0},
      {ObservationFeature::RemainingUses, 255.0}};
  return feature_normalizations;
}

// For backward compatibility, provide macros that expand to function calls
// This allows existing code to work without modification
#define FeatureNames GetFeatureNames()
#define FeatureNormalizations GetFeatureNormalizations()

const float DEFAULT_INVENTORY_NORMALIZATION = 100.0;

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONSTANTS_HPP_
