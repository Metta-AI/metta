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

// This must match GridFeatureNames below!
enum GridFeature {
  AGENT = 0,                 // Matches "agent" at index 0
  AGENT_GROUP = 1,           // Matches "agent:group" at index 1
  HP = 2,                    // Matches "hp" at index 2
  AGENT_FROZEN = 3,          // Matches "agent:frozen" at index 3
  AGENT_ORIENTATION = 4,     // Matches "agent:orientation" at index 4
  AGENT_COLOR = 5,           // Matches "agent:color" at index 5
  AGENT_INV_ORE_RED = 6,     // Matches "agent:inv:ore.red" at index 6
  AGENT_INV_ORE_BLUE = 7,    // Matches "agent:inv:ore.blue" at index 7
  AGENT_INV_ORE_GREEN = 8,   // Matches "agent:inv:ore.green" at index 8
  AGENT_INV_BATTERY = 9,     // Matches "agent:inv:battery" at index 9
  AGENT_INV_HEART = 10,      // Matches "agent:inv:heart" at index 10
  AGENT_INV_ARMOR = 11,      // Matches "agent:inv:armor" at index 11
  AGENT_INV_LASER = 12,      // Matches "agent:inv:laser" at index 12
  AGENT_INV_BLUEPRINT = 13,  // Matches "agent:inv:blueprint" at index 13
  WALL = 14,                 // Matches "wall" at index 14
  SWAPPABLE = 15,            // Matches "swappable" at index 15
  MINE = 16,                 // Matches "mine" at index 16
  COLOR = 17,                // Matches "color" at index 17
  CONVERTING = 18,           // Matches "converting" at index 18
  INV_ORE_RED = 19,          // Matches "inv:ore.red" at index 19
  INV_ORE_BLUE = 20,         // Matches "inv:ore.blue" at index 20
  INV_ORE_GREEN = 21,        // Matches "inv:ore.green" at index 21
  INV_BATTERY = 22,          // Matches "inv:battery" at index 22
  INV_HEART = 23,            // Matches "inv:heart" at index 23
  INV_ARMOR = 24,            // Matches "inv:armor" at index 24
  INV_LASER = 25,            // Matches "inv:laser" at index 25
  INV_BLUEPRINT = 26,        // Matches "inv:blueprint" at index 26
  GENERATOR = 27,            // Matches "generator" at index 27
  ALTAR = 28,                // Matches "altar" at index 28
  ARMORY = 29,               // Matches "armory" at index 29
  LASERY = 30,               // Matches "lasery" at index 30
  LAB = 31,                  // Matches "lab" at index 31
  FACTORY = 32,              // Matches "factory" at index 32
  TEMPLE = 33,               // Matches "temple" at index 33
  // Grid feature count - always keep this last
  COUNT  // Should equal the number of enum values
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