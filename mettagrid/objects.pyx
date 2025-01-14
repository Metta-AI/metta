# distutils: language=c++

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from mettagrid.grid_object cimport GridObject, GridObjectId

cdef vector[string] ObjectTypeNames = <vector[string]>[
    "agent",
    "wall",
    "generator",
    "converter",
    "altar"
]

cdef vector[string] InventoryItemNames = <vector[string]>[
    "r1",
    "r2",
    "r3"
]

ObjectLayers = {
    ObjectType.AgentT: GridLayer.Agent_Layer,
    ObjectType.WallT: GridLayer.Object_Layer,
    ObjectType.GeneratorT: GridLayer.Object_Layer,
    ObjectType.ConverterT: GridLayer.Object_Layer,
    ObjectType.AltarT: GridLayer.Object_Layer,
}
