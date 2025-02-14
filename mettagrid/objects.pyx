# distutils: language=c++

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef vector[string] ObjectTypeNames = <vector[string]>[
    "agent",
    "wall",
    "mine",
    "generator",
    "altar",
    "armory",
    "lasery",
    "lab",
    "factory",
    "temple"
]

cdef vector[string] InventoryItemNames = <vector[string]>[
    "ore",
    "battery",
    "heart",
    "armor",
    "laser",
    "blueprint"
]

ObjectLayers = {
    ObjectType.AgentT: GridLayer.Agent_Layer,
    ObjectType.WallT: GridLayer.Object_Layer,
    ObjectType.MineT: GridLayer.Object_Layer,
    ObjectType.GeneratorT: GridLayer.Object_Layer,
    ObjectType.AltarT: GridLayer.Object_Layer,
    ObjectType.ArmoryT: GridLayer.Object_Layer,
    ObjectType.LaseryT: GridLayer.Object_Layer,
    ObjectType.LabT: GridLayer.Object_Layer,
    ObjectType.FactoryT: GridLayer.Object_Layer,
    ObjectType.TempleT: GridLayer.Object_Layer,
}
