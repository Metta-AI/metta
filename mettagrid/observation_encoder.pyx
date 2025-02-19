import numpy as np
import gymnasium as gym

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from mettagrid.grid_object cimport GridObject, GridObjectId
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType

from mettagrid.objects.constants cimport ObjectType
from mettagrid.objects.mine cimport Mine
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.reset_handler cimport ResetHandler
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.lab cimport Lab
from mettagrid.objects.factory cimport Factory
from mettagrid.objects.temple cimport Temple
from mettagrid.objects.armory cimport Armory
from mettagrid.objects.lasery cimport Lasery
from mettagrid.objects.usable cimport Usable

cdef class MettaObservationEncoder(ObservationEncoder):
    cpdef obs_np_type(self):
        return np.uint8

    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)
        self._type_feature_names.resize(ObjectType.Count)
        features = []

        self._type_feature_names[ObjectType.AgentT] = Agent.feature_names()
        self._type_feature_names[ObjectType.WallT] = Wall.feature_names()
        self._type_feature_names[ObjectType.MineT] = Mine.feature_names()
        self._type_feature_names[ObjectType.GeneratorT] = Generator.feature_names()
        self._type_feature_names[ObjectType.AltarT] = Altar.feature_names()
        self._type_feature_names[ObjectType.ArmoryT] = Armory.feature_names()
        self._type_feature_names[ObjectType.LaseryT] = Lasery.feature_names()
        self._type_feature_names[ObjectType.LabT] = Lab.feature_names()
        self._type_feature_names[ObjectType.FactoryT] = Factory.feature_names()
        self._type_feature_names[ObjectType.TempleT] = Temple.feature_names()

        for type_id in range(ObjectType.Count):
            self._offsets[type_id] = len(features)
            features.extend(self._type_feature_names[type_id])
        self._feature_names = features

        print(self._type_feature_names)
        print(self._offsets)

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, self._offsets[obj._type_id])

    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset):
        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.MineT:
            (<Mine*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.GeneratorT:
            (<Generator*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.AltarT:
            (<Altar*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.ArmoryT:
            (<Armory*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.LaseryT:
            (<Lasery*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.LabT:
            (<Lab*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.FactoryT:
            (<Factory*>obj).obs(&obs[offset])
        elif obj._type_id == ObjectType.TempleT:
            (<Temple*>obj).obs(&obs[offset])
        else:
            printf("Encoding object of unknown type: %d\n", obj._type_id)

    cdef vector[string] feature_names(self):
        return self._feature_names

cdef class MettaCompactObservationEncoder(MettaObservationEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._num_features = 0
        for type_id in range(ObjectType.Count):
            self._num_features = max(self._num_features, len(self._type_feature_names[type_id]))

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, 0)
        obs[0] = obj._type_id + 1


    cpdef observation_space(self):
        type_info = np.iinfo(self.obs_np_type())

        return gym.spaces.Box(
                    low=type_info.min, high=type_info.max,
                    shape=(
                        self._num_features,
                        self._obs_height, self._obs_width),
            dtype=self.obs_np_type()
        )
