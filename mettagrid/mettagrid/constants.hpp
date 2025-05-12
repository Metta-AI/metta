#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "types.hpp"

enum Events {
  FinishConverting = 0,
  CoolDown = 1
};

enum GridLayer {
  Agent_Layer = 0,
  Object_Layer = 1
};

enum Orientation {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3
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
  InventoryCount = 8
};

// These should be const global variables
const std::vector<std::string> InventoryItemNames =
    {"ore.red", "ore.blue", "ore.green", "battery", "heart", "armor", "laser", "blueprint"};

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

// Action types enum - must match the order in which action handlers are registered
enum ActionType {
  PutRecipeItems = 0,
  GetOutput = 1,
  Noop = 2,
  Move = 3,
  Rotate = 4,
  Attack = 5,
  AttackNearest = 6,
  Swap = 7,
  ChangeColor = 8,
  ActionCount = 9
};

// Action names - must match the enum indices exactly
const std::vector<std::string> ActionTypeNames =
    {"put_recipe_items", "get_output", "noop", "move", "rotate", "attack", "attack_nearest", "swap", "change_color"};

// Maximum argument values for each action type
const std::vector<uint8_t> ActionMaxArgs = {
    0,  // PutRecipeItems
    0,  // GetOutput
    0,  // Noop
    3,  // Move (Up, Down, Left, Right)
    3,  // Rotate (Up, Down, Left, Right)
    3,  // Attack (direction)
    0,  // AttackNearest
    3,  // Swap (direction)
    2   // ChangeColor (color values)
};

enum GridFeature {
  // Basic object features
  HP = 0,
  HAS_INVENTORY = 1,

  // Wall features
  WALL = 2,
  SWAPPABLE = 3,

  // Agent features
  AGENT = 4,
  AGENT_GROUP = 5,
  AGENT_FROZEN = 6,
  AGENT_ORIENTATION = 7,
  AGENT_COLOR = 8,

  // Agent inventory features
  AGENT_INV_ORE_RED = 9,
  AGENT_INV_ORE_BLUE = 10,
  AGENT_INV_ORE_GREEN = 11,
  AGENT_INV_BATTERY = 12,
  AGENT_INV_HEART = 13,
  AGENT_INV_ARMOR = 14,
  AGENT_INV_LASER = 15,
  AGENT_INV_BLUEPRINT = 16,

  // General inventory features
  INV_ORE_RED = 17,
  INV_ORE_BLUE = 18,
  INV_ORE_GREEN = 19,
  INV_BATTERY = 20,
  INV_HEART = 21,
  INV_ARMOR = 22,
  INV_LASER = 23,
  INV_BLUEPRINT = 24,

  // Converter features
  COLOR = 25,
  CONVERTING = 26,

  // Object type features (one entry for each object type)
  AGENT_TYPE = 27,
  WALL_TYPE = 28,
  MINE_TYPE = 29,
  GENERATOR_TYPE = 30,
  ALTAR_TYPE = 31,
  ARMORY_TYPE = 32,
  LASERY_TYPE = 33,
  LAB_TYPE = 34,
  FACTORY_TYPE = 35,
  TEMPLE_TYPE = 36,
  CONVERTER_TYPE = 37,

  // Grid feature count - always keep this last
  COUNT = 38
};

// Feature names for debugging and display - must match enum indices exactly
const std::vector<std::string> GridFeatureNames = {
    "agent",                // 0
    "agent:group",          // 1
    "hp",                   // 2
    "agent:frozen",         // 3
    "agent:orientation",    // 4
    "agent:color",          // 5
    "agent:inv:ore.red",    // 6
    "agent:inv:ore.blue",   // 7
    "agent:inv:ore.green",  // 8
    "agent:inv:battery",    // 9
    "agent:inv:heart",      // 10
    "agent:inv:armor",      // 11
    "agent:inv:laser",      // 12
    "agent:inv:blueprint",  // 13
    "wall",                 // 14
    "swappable",            // 15
    "mine",                 // 16
    "color",                // 17
    "converting",           // 18
    "inv:ore.red",          // 19
    "inv:ore.blue",         // 20
    "inv:ore.green",        // 21
    "inv:battery",          // 22
    "inv:heart",            // 23
    "inv:armor",            // 24
    "inv:laser",            // 25
    "inv:blueprint",        // 26
    "generator",            // 27
    "altar",                // 28
    "armory",               // 29
    "lasery",               // 30
    "lab",                  // 31
    "factory",              // 32
    "temple",               // 33
};

#endif