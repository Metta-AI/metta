# distutils: language=c++

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from puffergrid.grid_object cimport GridObject, GridObjectId

cdef vector[string] ObjectTypeNames = [
    "agent",
    "wall",
    "generator",
    "converter",
    "altar"
]

cdef vector[string] InventoryItemNames = [
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

cdef class MettaObservationEncoder(ObservationEncoder):
    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)

        features = []
        self._offsets[ObjectType.AgentT] = 0
        features.extend(Agent.feature_names())

        self._offsets[ObjectType.WallT] = len(features)
        features.extend(Wall.feature_names())

        self._offsets[ObjectType.GeneratorT] = len(features)
        features.extend(Generator.feature_names())

        self._offsets[ObjectType.ConverterT] = len(features)
        features.extend(Converter.feature_names())

        self._offsets[ObjectType.AltarT] = len(features)
        features.extend(Altar.feature_names())

        self._feature_names = features

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).obs(obs[self._offsets[ObjectType.AgentT]:])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).obs(obs[self._offsets[ObjectType.WallT]:])
        elif obj._type_id == ObjectType.GeneratorT:
            (<Generator*>obj).obs(obs[self._offsets[ObjectType.GeneratorT]:])
        elif obj._type_id == ObjectType.ConverterT:
            (<Converter*>obj).obs(obs[self._offsets[ObjectType.ConverterT]:])
        elif obj._type_id == ObjectType.AltarT:
            (<Altar*>obj).obs(obs[self._offsets[ObjectType.AltarT]:])
        else:
            printf("Encoding object of unknown type: %d\n", obj._type_id)

    cdef vector[string] feature_names(self):
        return self._feature_names
