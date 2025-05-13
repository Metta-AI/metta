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

const std::vector<std::string> ObjectTypeNames =
    {"agent", "wall", "mine", "generator", "altar", "armory", "lasery", "lab", "factory", "temple", "converter"};

enum InventoryItem {
  ore_red = 0,
  ore_blue = 1,
  ore_green = 2,
  battery = 3,
  heart = 4,
  armor = 5,
  laser = 6,
  blueprint = 7,
  stub_0 = 8,
  stub_1 = 9,
  stub_2 = 10,
  stub_3 = 11,
  stub_4 = 12,
  stub_5 = 13,
  stub_6 = 14,
  stub_7 = 15,
  stub_8 = 16,
  stub_9 = 17,
  stub_10 = 18,
  stub_11 = 19,
  stub_12 = 20,
  stub_13 = 21,
  stub_14 = 22,
  stub_15 = 23,
  stub_16 = 24,
  stub_17 = 25,
  stub_18 = 26,
  stub_19 = 27,
  stub_20 = 28,
  stub_21 = 29,
  stub_22 = 30,
  stub_23 = 31,
  stub_24 = 32,
  InventoryCount = 33
};
const std::vector<std::string> InventoryItemNames =
    {"ore.red", "ore.blue", "ore.green", "battery", "heart", "armor", "laser", "blueprint",
     "stub.0", "stub.1", "stub.2", "stub.3", "stub.4", "stub.5", "stub.6", "stub.7", 
     "stub.8", "stub.9", "stub.10", "stub.11", "stub.12", "stub.13", "stub.14", "stub.15",
     "stub.16", "stub.17", "stub.18", "stub.19", "stub.20", "stub.21", "stub.22", "stub.23", "stub.24"};

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

#endif
