import numpy as np
import gymnasium as gym

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

from mettagrid.grid_object cimport GridObject, ObsType

from mettagrid.objects.constants cimport ObjectType

from mettagrid.objects.agent cimport Agent
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.armory cimport Armory
from mettagrid.objects.converter cimport Converter
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
            for i in range(len(self._type_feature_names[type_id])):
                self._offsets[type_id].push_back(len(features))
                features.append(self._type_feature_names[type_id][i])
        self._feature_names = features

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, self._offsets[obj._type_id])

    cdef _encode(self, GridObject *obj, ObsType[:] obs, vector[unsigned int] offsets):
        obj.obs(&obs[0], offsets)

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

cdef class SemiCompactObservationEncoder(ObservationEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._offsets.resize(ObjectType.Count)
        self._type_feature_names.resize(ObjectType.Count)
        # Generate an offset for each unique feature name.
        cdef map[string, int] features
        cdef vector[string] feature_names
        for type_id in range(ObjectType.Count):
            for i in range(len(self._type_feature_names[type_id])):
                if features.count(self._type_feature_names[type_id][i]) == 0:
                    features[self._type_feature_names[type_id][i]] = features.size()
                    feature_names.push_back(self._type_feature_names[type_id][i])
        # Set the offset for each feature, using the global offsets.
        for type_id in range(ObjectType.Count):
            for i in range(len(self._type_feature_names[type_id])):
                self._offsets[type_id][i] = features[self._type_feature_names[type_id][i]]

        self._feature_names = feature_names
