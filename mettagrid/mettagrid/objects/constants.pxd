# distutils: language=c++
# cython: warn.undeclared=False

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport TypeId

cdef extern from "constants.hpp":
    cdef enum Events:
        FinishConverting = 0

    cdef enum GridLayer:
        Agent_Layer = 0
        Object_Layer = 1

    cdef enum ObjectType:
        AgentT = 0
        WallT = 1
        MineT = 2
        GeneratorT = 3
        AltarT = 4
        ArmoryT = 5
        LaseryT = 6
        LabT = 7
        FactoryT = 8
        TempleT = 9
        GenericConverterT = 10
        Count = 11

    cdef enum InventoryItem:
        ore = 0
        battery = 1
        heart = 2
        armor = 3
        laser = 4
        blueprint = 5
        InventoryCount = 6

    cdef vector[string] InventoryItemNames
    cdef vector[string] ObjectTypeNames
    cdef map[TypeId, GridLayer] ObjectLayers
