# distutils: language=c++

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from puffergrid.grid_object cimport GridObject, GridObjectId

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

cdef class MettaObservationEncoder(ObservationEncoder):
    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)
        self._type_feature_names.resize(ObjectType.Count)
        features = []

        self._type_feature_names[ObjectType.AgentT] = Agent.feature_names()
        self._type_feature_names[ObjectType.WallT] = Wall.feature_names()
        self._type_feature_names[ObjectType.GeneratorT] = Generator.feature_names()
        self._type_feature_names[ObjectType.ConverterT] = Converter.feature_names()
        self._type_feature_names[ObjectType.AltarT] = Altar.feature_names()

        for type_id in range(ObjectType.Count):
            self._offsets[type_id] = len(features)
            features.extend(self._type_feature_names[type_id])
        self._feature_names = features

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, self._offsets[obj._type_id])

    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset):
        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.GeneratorT:
            (<Generator*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.ConverterT:
            (<Converter*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.AltarT:
            (<Altar*>obj).obs(obs[offset:])
        else:
            printf("Encoding object of unknown type: %d\n", obj._type_id)

    cdef vector[string] feature_names(self):
        return self._feature_names
