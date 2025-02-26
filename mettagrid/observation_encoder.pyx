import numpy as np
import gymnasium as gym

from libcpp.string cimport string
from libcpp.vector cimport vector

from mettagrid.grid_object cimport GridObject, ObsType

from mettagrid.objects.constants cimport ObjectType

from mettagrid.objects.agent cimport Agent
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.armory cimport Armory
from mettagrid.objects.factory cimport Factory
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.lab cimport Lab
from mettagrid.objects.lasery cimport Lasery
from mettagrid.objects.mine cimport Mine
from mettagrid.objects.temple cimport Temple
from mettagrid.objects.wall cimport Wall

cdef class ObservationEncoder:
    cpdef obs_np_type(self):
        return np.uint8
    
    cdef init(self, unsigned int obs_width, unsigned int obs_height):
        self._obs_width = obs_width
        self._obs_height = obs_height

    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)
        self._type_feature_names.resize(ObjectType.Count)
        features = []

        self._type_feature_names[ObjectType.AgentT] = Agent.feature_names()
        self._type_feature_names[ObjectType.AltarT] = Altar.feature_names()
        self._type_feature_names[ObjectType.ArmoryT] = Armory.feature_names()
        self._type_feature_names[ObjectType.FactoryT] = Factory.feature_names()
        self._type_feature_names[ObjectType.GeneratorT] = Generator.feature_names()
        self._type_feature_names[ObjectType.LabT] = Lab.feature_names()
        self._type_feature_names[ObjectType.LaseryT] = Lasery.feature_names()
        self._type_feature_names[ObjectType.MineT] = Mine.feature_names()
        self._type_feature_names[ObjectType.TempleT] = Temple.feature_names()
        self._type_feature_names[ObjectType.WallT] = Wall.feature_names()

        for type_id in range(ObjectType.Count):
            self._offsets[type_id] = len(features)
            features.extend(self._type_feature_names[type_id])
        self._feature_names = features

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, self._offsets[obj._type_id])

    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset):
        obj.obs(&obs[offset])

    cdef vector[string] feature_names(self):
        return self._feature_names
    
    cpdef observation_space(self):
        type_info = np.iinfo(self.obs_np_type())

        return gym.spaces.Box(
                    low=type_info.min, high=type_info.max,
                    shape=(
                        len(self.feature_names()),
                        self._obs_height, self._obs_width),
            dtype=self.obs_np_type()
        )

cdef class CompactObservationEncoder(ObservationEncoder):
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
