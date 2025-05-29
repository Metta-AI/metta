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
  TypeId = 1,
  Group = 2,
  Hp = 3,
  Frozen = 4,
  Orientation = 5,
  Color = 6,
  ConvertingOrCoolingDown = 7,
  Swappable = 8,
  ObservationFeatureCount
};
}  // namespace ObservationFeature

const uint8_t InventoryFeatureOffset = 100;

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

constexpr std::array<const char*, InventoryItemCount> InventoryItemNamesArray = {
    {"ore.red", "ore.blue", "ore.green", "battery.red", "battery.blue", "battery.green", "heart", "armor", "laser", "blueprint"}};

const std::vector<std::string> InventoryItemNames(InventoryItemNamesArray.begin(), InventoryItemNamesArray.end());

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
