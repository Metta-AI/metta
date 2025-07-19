#ifndef OBJECTS_CONSTANTS_HPP_
#define OBJECTS_CONSTANTS_HPP_

#include <map>
#include <string>
#include <vector>

#include "types.hpp"

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
constexpr ObservationType Hp = 2;
constexpr ObservationType Frozen = 3;
constexpr ObservationType Orientation = 4;
constexpr ObservationType Color = 5;
constexpr ObservationType ConvertingOrCoolingDown = 6;
constexpr ObservationType Swappable = 7;
constexpr ObservationType EpisodeCompletionPct = 8;
constexpr ObservationType LastAction = 9;
constexpr ObservationType LastActionArg = 10;
constexpr ObservationType LastReward = 11;
constexpr ObservationType Glyph = 12;
constexpr ObservationType ResourceRewards = 13;
constexpr ObservationType ObservationFeatureCount = 14;
}  // namespace ObservationFeature

const ObservationType InventoryFeatureOffset = ObservationFeature::ObservationFeatureCount;

// Recipe inputs start at offset 50 to avoid collision with inventory items
// Typically there are fewer than 20 inventory items, so this leaves plenty of room
const ObservationType RecipeInputFeatureOffset = 50;

const std::map<ObservationType, std::string> FeatureNames = {
    {ObservationFeature::TypeId, "type_id"},
    {ObservationFeature::Group, "agent:group"},
    {ObservationFeature::Hp, "hp"},
    {ObservationFeature::Frozen, "agent:frozen"},
    {ObservationFeature::Orientation, "agent:orientation"},
    {ObservationFeature::Color, "agent:color"},
    {ObservationFeature::ConvertingOrCoolingDown, "converting"},
    {ObservationFeature::Swappable, "swappable"},
    {ObservationFeature::EpisodeCompletionPct, "episode_completion_pct"},
    {ObservationFeature::LastAction, "last_action"},
    {ObservationFeature::LastActionArg, "last_action_arg"},
    {ObservationFeature::LastReward, "last_reward"},
    {ObservationFeature::Glyph, "agent:glyph"},
    {ObservationFeature::ResourceRewards, "resource_rewards"}};

// ##ObservationNormalization
// These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
// but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
// These should move to configuration "soon". E.g., by 2025-06-10.
const std::map<ObservationType, float> FeatureNormalizations = {
    {ObservationFeature::LastAction, 10.0},
    {ObservationFeature::LastActionArg, 10.0},
    {ObservationFeature::EpisodeCompletionPct, 255.0},
    {ObservationFeature::LastReward, 100.0},
    {ObservationFeature::TypeId, 1.0},
    {ObservationFeature::Group, 10.0},
    {ObservationFeature::Hp, 30.0},
    {ObservationFeature::Frozen, 1.0},
    {ObservationFeature::Orientation, 1.0},
    {ObservationFeature::Color, 255.0},
    {ObservationFeature::ConvertingOrCoolingDown, 1.0},
    {ObservationFeature::Swappable, 1.0},
    {ObservationFeature::Glyph, 255.0},
    {ObservationFeature::ResourceRewards, 255.0},
};

const float DEFAULT_INVENTORY_NORMALIZATION = 100.0;

#endif  // OBJECTS_CONSTANTS_HPP_
