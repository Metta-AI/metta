#ifndef METTAGRID_METTAGRID_OBJECTS_CONSTANTS_HPP_
#define METTAGRID_METTAGRID_OBJECTS_CONSTANTS_HPP_

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
  MineT = 2,
  GeneratorT = 3,
  AltarT = 4,
  ArmoryT = 5,
  LaseryT = 6,
  LabT = 7,
  FactoryT = 8,
  TempleT = 9,
  GenericConverterT = 10,
  ObjectTypeCount
};

constexpr std::array<const char*, ObjectTypeCount> ObjectTypeNamesArray = {
    {"agent", "wall", "mine", "generator", "altar", "armory", "lasery", "lab", "factory", "temple", "converter"}};

const std::vector<std::string> ObjectTypeNames(ObjectTypeNamesArray.begin(), ObjectTypeNamesArray.end());

enum InventoryItem {
  // These are "ore.red", etc everywhere else. They're differently named here because
  // of enum naming limitations.
  ore_red = 0,
  ore_blue = 1,
  ore_green = 2,
  battery_red = 3,
  battery_blue = 4,
  battery_green = 5,
  heart = 6,
  armor = 7,
  laser = 8,
  blueprint = 9,
  InventoryItemCount
};

constexpr std::array<const char*, InventoryItemCount> InventoryItemNamesArray = {{"ore.red",
                                                                                  "ore.blue",
                                                                                  "ore.green",
                                                                                  "battery.red",
                                                                                  "battery.blue",
                                                                                  "battery.green",
                                                                                  "heart",
                                                                                  "armor",
                                                                                  "laser",
                                                                                  "blueprint"}};

const std::vector<std::string> InventoryItemNames(InventoryItemNamesArray.begin(), InventoryItemNamesArray.end());

constexpr std::array<const char*, ObservationFeature::ObservationFeatureCount> ObservationFeatureNamesArray = {
    {"type_id",
     "agent:group",
     "hp",
     "agent:frozen",
     "agent:orientation",
     "agent:color",
     "converting",
     "swappable",
     "episode_completion_pct",
     "last_action",
     "last_action_arg",
     "last_reward"}};

const std::vector<std::string> ObservationFeatureNames = []() {
  std::vector<std::string> names;
  names.reserve(ObservationFeatureNamesArray.size() + InventoryItemNamesArray.size());
  names.insert(names.end(), ObservationFeatureNamesArray.begin(), ObservationFeatureNamesArray.end());
  for (const auto& name : InventoryItemNames) {
    names.push_back("inv:" + name);
  }
  return names;
}();

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
    {InventoryFeatureOffset + InventoryItem::ore_red, 100.0},
    {InventoryFeatureOffset + InventoryItem::ore_blue, 100.0},
    {InventoryFeatureOffset + InventoryItem::ore_green, 100.0},
    {InventoryFeatureOffset + InventoryItem::battery_red, 100.0},
    {InventoryFeatureOffset + InventoryItem::battery_blue, 100.0},
    {InventoryFeatureOffset + InventoryItem::battery_green, 100.0},
    {InventoryFeatureOffset + InventoryItem::heart, 100.0},
    {InventoryFeatureOffset + InventoryItem::laser, 100.0},
    {InventoryFeatureOffset + InventoryItem::armor, 100.0},
    {InventoryFeatureOffset + InventoryItem::blueprint, 100.0},
};

const float DEFAULT_NORMALIZATION = 1.0;

const std::map<TypeId, GridLayer> ObjectLayers = {{ObjectType::AgentT, GridLayer::Agent_Layer},
                                                  {ObjectType::WallT, GridLayer::Object_Layer},
                                                  {ObjectType::MineT, GridLayer::Object_Layer},
                                                  {ObjectType::GeneratorT, GridLayer::Object_Layer},
                                                  {ObjectType::AltarT, GridLayer::Object_Layer},
                                                  {ObjectType::ArmoryT, GridLayer::Object_Layer},
                                                  {ObjectType::LaseryT, GridLayer::Object_Layer},
                                                  {ObjectType::LabT, GridLayer::Object_Layer},
                                                  {ObjectType::FactoryT, GridLayer::Object_Layer},
                                                  {ObjectType::TempleT, GridLayer::Object_Layer}};

#endif  // METTAGRID_METTAGRID_OBJECTS_CONSTANTS_HPP_
