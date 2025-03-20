#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <map>
#include <vector>
#include <string>

#include "../grid_object.hpp"

enum Events {
    FinishConverting = 0,
    CoolDown = 1
};

enum GridLayer {
    Agent_Layer = 0,
    Object_Layer = 1
};

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

enum InventoryItem {
    ore_red = 0,
    ore_blue =1,
    ore_green = 2,
    battery = 3,
    heart = 4,
    armor = 5,
    laser = 6,
    blueprint = 7,
    InventoryCount = 8
};

// These should be const, but we run into type inference issues with cython
std::vector<std::string> InventoryItemNames = {
    "ore.red",
    "ore.blue",
    "ore.green",
    "battery",
    "heart",
    "armor",
    "laser",
    "blueprint"
};

std::vector<std::string> ObjectTypeNames = {
    "agent",
    "wall",
    "block",
    "mine.red",
    "mine.blue",
    "mine.green",
    "generator.red",
    "generator.blue",
    "generator.green",
    "altar",
    "armory",
    "lasery",
    "lab",
    "factory",
    "temple",
    "converter"
};

std::map<TypeId, GridLayer> ObjectLayers = {
    {ObjectType::AgentT, GridLayer::Agent_Layer},
    {ObjectType::WallT, GridLayer::Object_Layer},
    {ObjectType::MineT, GridLayer::Object_Layer},
    {ObjectType::GeneratorT, GridLayer::Object_Layer},
    {ObjectType::AltarT, GridLayer::Object_Layer},
    {ObjectType::ArmoryT, GridLayer::Object_Layer},
    {ObjectType::LaseryT, GridLayer::Object_Layer},
    {ObjectType::LabT, GridLayer::Object_Layer},
    {ObjectType::FactoryT, GridLayer::Object_Layer},
    {ObjectType::TempleT, GridLayer::Object_Layer}
};

#endif
