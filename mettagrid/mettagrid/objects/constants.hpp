#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <map>
#include <string>
#include <vector>

#include "../grid_object.hpp"

enum Events {
  FinishConverting = 0,
  CoolDown = 1
};

enum GridLayer {
  Agent_Layer = 0,
  Object_Layer = 1
};

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
  Count = 11
};

std::vector<std::string> ObjectTypeNames =
    {"agent", "wall", "mine", "generator", "altar", "armory", "lasery", "lab", "factory", "temple", "converter"};

std::vector<std::string> ObjectTypeAscii = {"A", "#", "g", "c", "a", "r", "l", "b", "f", "t", "v"};

enum InventoryItem {
  ore_red = 0,
  ore_blue = 1,
  ore_green = 2,
  battery = 3,
  heart = 4,
  armor = 5,
  laser = 6,
  blueprint = 7,
  InventoryCount = 8
};

// These should be const, but we run into type inference issues with cython
std::vector<std::string> InventoryItemNames =
    {"ore.red", "ore.blue", "ore.green", "battery", "heart", "armor", "laser", "blueprint"};

std::map<TypeId, GridLayer> ObjectLayers = {{ObjectType::AgentT, GridLayer::Agent_Layer},
                                            {ObjectType::WallT, GridLayer::Object_Layer},
                                            {ObjectType::MineT, GridLayer::Object_Layer},
                                            {ObjectType::GeneratorT, GridLayer::Object_Layer},
                                            {ObjectType::AltarT, GridLayer::Object_Layer},
                                            {ObjectType::ArmoryT, GridLayer::Object_Layer},
                                            {ObjectType::LaseryT, GridLayer::Object_Layer},
                                            {ObjectType::LabT, GridLayer::Object_Layer},
                                            {ObjectType::FactoryT, GridLayer::Object_Layer},
                                            {ObjectType::TempleT, GridLayer::Object_Layer}};

#endif
