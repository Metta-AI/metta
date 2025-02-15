from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from mettagrid.grid_object cimport GridObject
from mettagrid.objects.constants cimport ObjectType, InventoryItem
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.converter cimport Converter
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
import numpy as np
import gymnasium as gym

cdef class MettaObservationEncoder(ObservationEncoder):
    cpdef obs_np_type(self):
        return np.uint8

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
            # We inline the agent observations since it's a pain to use memoryviews in cpp
            obs[offset] = 1
            obs[offset + 1] = (<Agent*>obj).group
            obs[offset + 2] = (<Agent*>obj).hp
            obs[offset + 3] = (<Agent*>obj).frozen
            obs[offset + 4] = (<Agent*>obj).energy
            obs[offset + 5] = (<Agent*>obj).orientation
            obs[offset + 6] = (<Agent*>obj).shield
            obs[offset + 7] = (<Agent*>obj).color
            # #InlineInventoryCount
            for i in range(3):
                obs[<int>(offset + 8 + i)] = (<Agent*>obj).inventory[i]
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.GeneratorT:
            (<Generator*>obj).obs(obs[offset:])
        elif obj._type_id == ObjectType.ConverterT:
            obs[offset] = 1
            obs[offset + 1] = (<Converter*>obj).prey_r1_output_energy
            obs[offset + 2] = (<Converter*>obj).predator_r1_output_energy
            obs[offset + 3] = (<Converter*>obj).predator_r2_output_energy
        elif obj._type_id == ObjectType.AltarT:
            (<Altar*>obj).obs(obs[offset:])
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
