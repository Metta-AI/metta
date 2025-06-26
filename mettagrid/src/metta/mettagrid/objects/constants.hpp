#ifndef OBJECTS_CONSTANTS_HPP_
#define OBJECTS_CONSTANTS_HPP_

#include <map>
#include <string>
#include <vector>

#include "../grid_object.hpp"

enum EventType {
  FinishConverting = 0,
  CoolDown = 1,
  EventTypeCount
};

enum GridLayer {
  Agent_Layer = 0,
  Object_Layer = 1,
  GridLayerCount
};

// We want empty tokens to be 0xff, since 0s are very natural numbers to have in the observations, and we want
// empty to be obviously different.
const uint8_t EmptyTokenByte = 0xff;

// Changing observation feature ids will break models that have
// been trained on the old feature ids.
// In the future, the string -> id mapping should be stored on a
// per-policy basis.
//
// NOTE: We use a namespace here to avoid naming collisions:
// - 'TypeId' conflicts with the typedef uint8_t TypeId in grid_object.hpp
// - 'Orientation' conflicts with the enum class Orientation defined above
// The namespace allows us to use these descriptive names without conflicts.
namespace ObservationFeature {
enum ObservationFeatureEnum : uint8_t {
  TypeId = 0,
  Group = 1,
  Hp = 2,
  Frozen = 3,
  Orientation = 4,
  Color = 5,
  ConvertingOrCoolingDown = 6,
  Swappable = 7,
  EpisodeCompletionPct = 8,
  LastAction = 9,
  LastActionArg = 10,
  LastReward = 11,
  ObservationFeatureCount
};
}  // namespace ObservationFeature

const uint8_t InventoryFeatureOffset = ObservationFeature::ObservationFeatureCount;

// There should be a one-to-one mapping between ObjectType and ObjectTypeNames.
// ObjectTypeName is mostly used for human-readability, but may be used as a key
// in config files, etc. Agents will be able to see an object's type_id.
//
// Note that ObjectType does _not_ have to correspond to an object's class (which
// is a C++ concept). In particular, multiple ObjectTypes may correspond to the
// same class.
enum ObjectType {
  AgentT = 0,
  WallT = 1,
  MineRedT = 2,
  MineBlueT = 3,
  MineGreenT = 4,
  GeneratorRedT = 5,
  GeneratorBlueT = 6,
  GeneratorGreenT = 7,
  AltarT = 8,
  ArmoryT = 9,
  LaseryT = 10,
  LabT = 11,
  FactoryT = 12,
  TempleT = 13,
  GenericConverterT = 14,
  ObjectTypeCount
};

constexpr std::array<const char*, ObjectTypeCount> ObjectTypeNamesArray = {{"agent",
                                                                            "wall",
                                                                            "mine_red",
                                                                            "mine_blue",
                                                                            "mine_green",
                                                                            "generator_red",
                                                                            "generator_blue",
                                                                            "generator_green",
                                                                            "altar",
                                                                            "armory",
                                                                            "lasery",
                                                                            "lab",
                                                                            "factory",
                                                                            "temple",
                                                                            "converter"}};

const std::vector<std::string> ObjectTypeNames(ObjectTypeNamesArray.begin(), ObjectTypeNamesArray.end());

const std::map<uint8_t, std::string> FeatureNames = {
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
    {ObservationFeature::LastReward, "last_reward"}};

// ##ObservationNormalization
// These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
// but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
// These should move to configuration "soon". E.g., by 2025-06-10.
const std::map<uint8_t, float> FeatureNormalizations = {
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
};

const float DEFAULT_INVENTORY_NORMALIZATION = 100.0;

const std::map<TypeId, GridLayer> ObjectLayers = {{ObjectType::AgentT, GridLayer::Agent_Layer},
                                                  {ObjectType::WallT, GridLayer::Object_Layer},
                                                  {ObjectType::MineRedT, GridLayer::Object_Layer},
                                                  {ObjectType::MineBlueT, GridLayer::Object_Layer},
                                                  {ObjectType::MineGreenT, GridLayer::Object_Layer},
                                                  {ObjectType::GeneratorRedT, GridLayer::Object_Layer},
                                                  {ObjectType::GeneratorBlueT, GridLayer::Object_Layer},
                                                  {ObjectType::GeneratorGreenT, GridLayer::Object_Layer},
                                                  {ObjectType::AltarT, GridLayer::Object_Layer},
                                                  {ObjectType::ArmoryT, GridLayer::Object_Layer},
                                                  {ObjectType::LaseryT, GridLayer::Object_Layer},
                                                  {ObjectType::LabT, GridLayer::Object_Layer},
                                                  {ObjectType::FactoryT, GridLayer::Object_Layer},
                                                  {ObjectType::TempleT, GridLayer::Object_Layer}};

#endif  // OBJECTS_CONSTANTS_HPP_
