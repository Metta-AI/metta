#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <map>
#include <vector>
#include <string>

#include "../grid_object.hpp"

enum Events {
    Reset = 0
};

enum GridLayer {
    Agent_Layer = 0,
    Object_Layer = 1
};

enum ObjectType {
    AgentT = 0,
    WallT = 1,
    GeneratorT = 2,
    ConverterT = 3,
    AltarT = 4,
    Count = 5,
    Resource_Count = 2
};

enum InventoryItem {
    r1 = 0,
    r2 = 1,
    r3 = 2,
    InventoryCount = 3
};

// These should be const, but we run into type inference issues with cython
std::vector<std::string> InventoryItemNames = {
    "r1",
    "r2", 
    "r3"
};

std::vector<std::string> ObjectTypeNames = {
    "agent",
    "wall",
    "generator", 
    "converter",
    "altar"
};

std::map<TypeId, GridLayer> ObjectLayers = {
    {ObjectType::AgentT, GridLayer::Agent_Layer},
    {ObjectType::WallT, GridLayer::Object_Layer},
    {ObjectType::GeneratorT, GridLayer::Object_Layer},
    {ObjectType::ConverterT, GridLayer::Object_Layer}, 
    {ObjectType::AltarT, GridLayer::Object_Layer}
};

#endif
